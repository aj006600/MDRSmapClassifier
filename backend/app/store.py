"""SQLite 儲存:工作的狀態、抓回來的每日市場資料,以及每支股票每個 Horizon 的結果。

結果是 Search 的日期、起點、參數與逐日預測,之後的執行從這裡讀回來沿用。
API 與執行工作的程序各自開連線,共用同一個資料庫檔案。
"""

import pickle
import sqlite3
from datetime import UTC, date, datetime
from enum import StrEnum
from pathlib import Path

import numpy as np
import pandas as pd

from app.market import DAILY_DTYPES
from app.runs import RunResult

# 依序套用的結構變更,已經套用到第幾個記在資料庫的 user_version。
# 只能在最後面新增;改動已經發佈的項目,已有的資料庫不會跟著改
_MIGRATIONS = [
    """
    CREATE TABLE jobs (
        id INTEGER PRIMARY KEY,
        stock_id TEXT NOT NULL,
        horizon INTEGER NOT NULL,
        status TEXT NOT NULL,
        stage TEXT,
        error TEXT,
        created_at TEXT NOT NULL,
        started_at TEXT,
        finished_at TEXT
    );
    CREATE TABLE predictions (
        stock_id TEXT NOT NULL,
        horizon INTEGER NOT NULL,
        date TEXT NOT NULL,
        direction INTEGER NOT NULL,
        actual INTEGER,
        PRIMARY KEY (stock_id, horizon, date)
    );
    """,
    # 存下每日市場資料與 Search 的參數。舊版的逐日預測沒有存下參數,無法沿用,
    # 所以一併刪掉,下次執行時重新做 Search
    """
    ALTER TABLE jobs ADD COLUMN as_of TEXT;
    ALTER TABLE jobs ADD COLUMN re_search INTEGER NOT NULL DEFAULT 0;
    CREATE TABLE searches (
        stock_id TEXT NOT NULL,
        horizon INTEGER NOT NULL,
        search_date TEXT NOT NULL,
        start TEXT NOT NULL,
        params BLOB NOT NULL,
        PRIMARY KEY (stock_id, horizon)
    );
    CREATE TABLE daily (
        stock_id TEXT NOT NULL,
        date TEXT NOT NULL,
        open REAL NOT NULL,
        high REAL NOT NULL,
        low REAL NOT NULL,
        close REAL NOT NULL,
        volume INTEGER NOT NULL,
        trading_value INTEGER NOT NULL,
        margin_balance INTEGER,
        foreign_shares INTEGER,
        trust_net_buy INTEGER,
        dealer_net_buy INTEGER,
        PRIMARY KEY (stock_id, date)
    );
    DELETE FROM predictions;
    """,
]

# 標準每日資料表的欄名,以逗號分隔,直接放進 SQL。
# DAILY_DTYPES 增減欄位時,daily 資料表要用新的 migration 跟著改,並清掉已存的資料:
# 快取只往後補抓,不會替已經存下的日子補上新欄位
_DAILY_COLUMN_LIST = ", ".join(DAILY_DTYPES)


class Status(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


_IN_PROGRESS = (Status.QUEUED, Status.RUNNING)
"""還沒結束的工作的狀態。"""


def connect(path: Path) -> sqlite3.Connection:
    """開一條連線,查詢結果的每一列可以用欄名取值。

    同一條連線只給一個請求或一個工作依序使用,但 FastAPI 會在不同的執行緒裡
    建立與使用它,所以關掉 sqlite3 的同執行緒檢查。
    """
    connection = sqlite3.connect(path, timeout=30, check_same_thread=False)
    connection.row_factory = sqlite3.Row
    return connection


def migrate(connection: sqlite3.Connection) -> None:
    """套用還沒套用的結構變更;已經是最新的結構時什麼都不做。"""
    applied = connection.execute("PRAGMA user_version").fetchone()[0]
    for version, script in enumerate(_MIGRATIONS[applied:], start=applied + 1):
        connection.executescript(
            f"BEGIN; {script} PRAGMA user_version = {version}; COMMIT;"
        )


def create_job(
    connection: sqlite3.Connection,
    stock_id: str,
    horizon: int,
    as_of: date,
    re_search: bool,
) -> int:
    """新增一個以 as_of 為準的排隊中工作,回傳它的 id。

    re_search 為 True 的工作不沿用已存的參數,從新的起點重新做 Search。
    這支股票這個 Horizon 已經有排隊中或執行中的工作時,不新增,回傳那個工作的 id,
    不論那個工作有沒有要求重新搜尋。
    """
    with connection:
        # 先鎖住寫入再查:兩個同時發起的請求才不會都查不到、各新增一個
        connection.execute("BEGIN IMMEDIATE")
        existing = connection.execute(
            "SELECT id FROM jobs"
            " WHERE stock_id = ? AND horizon = ? AND status IN (?, ?)",
            (stock_id, horizon, *_IN_PROGRESS),
        ).fetchone()
        if existing:
            return existing["id"]
        cursor = connection.execute(
            "INSERT INTO jobs (stock_id, horizon, status, created_at, as_of, re_search)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (stock_id, horizon, Status.QUEUED, _now(), as_of.isoformat(), re_search),
        )
    return cursor.lastrowid


def get_job(connection: sqlite3.Connection, job_id: int) -> dict | None:
    row = connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    return dict(row) if row else None


def find_succeeded_job(
    connection: sqlite3.Connection, stock_id: str, horizon: int, as_of: date
) -> dict | None:
    """回傳這支股票這個 Horizon 以 as_of 為準、最近一個成功的工作;沒有時是 None。"""
    row = connection.execute(
        "SELECT * FROM jobs WHERE stock_id = ? AND horizon = ? AND as_of = ?"
        " AND status = ? ORDER BY id DESC LIMIT 1",
        (stock_id, horizon, as_of.isoformat(), Status.SUCCEEDED),
    ).fetchone()
    return dict(row) if row else None


def list_active_jobs(connection: sqlite3.Connection) -> list[dict]:
    """回傳排隊中與執行中的工作,依執行的順序排列。"""
    rows = connection.execute(
        "SELECT * FROM jobs WHERE status IN (?, ?) ORDER BY id", _IN_PROGRESS
    )
    return [dict(row) for row in rows]


def claim_next_job(connection: sqlite3.Connection) -> int | None:
    """把最早發起的排隊中工作標為執行中,回傳它的 id。

    沒有排隊中的工作時回傳 None。
    """
    with connection:
        # 先 fetchall 再 commit:RETURNING 的結果沒讀完時,這個 statement 還沒結束
        rows = connection.execute(
            "UPDATE jobs SET status = ?, started_at = ?"
            " WHERE id = (SELECT id FROM jobs WHERE status = ? ORDER BY id LIMIT 1)"
            " RETURNING id",
            (Status.RUNNING, _now(), Status.QUEUED),
        ).fetchall()
    return rows[0]["id"] if rows else None


def set_stage(connection: sqlite3.Connection, job_id: int, stage: str) -> None:
    with connection:
        connection.execute("UPDATE jobs SET stage = ? WHERE id = ?", (stage, job_id))


def fail_running_jobs(connection: sqlite3.Connection, error: str) -> None:
    """把所有執行中的工作標為失敗,失敗原因都是 error。"""
    with connection:
        connection.execute(
            "UPDATE jobs SET status = ?, error = ?, finished_at = ? WHERE status = ?",
            (Status.FAILED, error, _now(), Status.RUNNING),
        )


def fail_job(connection: sqlite3.Connection, job_id: int, error: str) -> None:
    with connection:
        connection.execute(
            "UPDATE jobs SET status = ?, error = ?, finished_at = ? WHERE id = ?",
            (Status.FAILED, error, _now(), job_id),
        )


def finish_job(connection: sqlite3.Connection, job_id: int, result: RunResult) -> None:
    """存下工作的結果,並把工作標為成功。

    這支股票這個 Horizon 原本存的結果會整個換掉。
    """
    job = get_job(connection, job_id)
    key = (job["stock_id"], job["horizon"])
    rows = [
        (
            *key,
            day.date().isoformat(),
            int(row.direction),
            None if pd.isna(row.actual) else int(row.actual),
        )
        for day, row in result.days.iterrows()
    ]
    with connection:
        connection.execute(
            "INSERT OR REPLACE INTO searches"
            " (stock_id, horizon, search_date, start, params) VALUES (?, ?, ?, ?, ?)",
            (
                *key,
                result.search_date.isoformat(),
                result.start.isoformat(),
                pickle.dumps(result.params),
            ),
        )
        connection.execute(
            "DELETE FROM predictions WHERE stock_id = ? AND horizon = ?", key
        )
        connection.executemany(
            "INSERT INTO predictions (stock_id, horizon, date, direction, actual)"
            " VALUES (?, ?, ?, ?, ?)",
            rows,
        )
        connection.execute(
            "UPDATE jobs SET status = ?, finished_at = ? WHERE id = ?",
            (Status.SUCCEEDED, _now(), job_id),
        )


def get_predictions(
    connection: sqlite3.Connection, stock_id: str, horizon: int
) -> list[dict]:
    """回傳逐日預測,由舊到新排列;還沒有結果時是空的。

    每一筆有 date、direction 與 actual;actual 還沒揭曉時是 None。
    """
    rows = connection.execute(
        "SELECT date, direction, actual FROM predictions"
        " WHERE stock_id = ? AND horizon = ? ORDER BY date",
        (stock_id, horizon),
    )
    return [dict(row) for row in rows]


def get_search_date(
    connection: sqlite3.Connection, stock_id: str, horizon: int
) -> date | None:
    """回傳這支股票這個 Horizon 最近一次做 Search 的日期;還沒有結果時是 None。"""
    row = connection.execute(
        "SELECT search_date FROM searches WHERE stock_id = ? AND horizon = ?",
        (stock_id, horizon),
    ).fetchone()
    return date.fromisoformat(row["search_date"]) if row else None


def get_run(
    connection: sqlite3.Connection, stock_id: str, horizon: int
) -> RunResult | None:
    """回傳這支股票這個 Horizon 最近一次成功執行的結果;還沒有結果時是 None。"""
    search = connection.execute(
        "SELECT search_date, start, params FROM searches"
        " WHERE stock_id = ? AND horizon = ?",
        (stock_id, horizon),
    ).fetchone()
    if search is None:
        return None
    rows = get_predictions(connection, stock_id, horizon)
    days = pd.DataFrame(
        {
            "direction": pd.Series([row["direction"] for row in rows], dtype="int64"),
            "actual": pd.Series([row["actual"] for row in rows], dtype="Int64"),
        }
    )
    days.index = pd.DatetimeIndex([row["date"] for row in rows], name="date")
    return RunResult(
        search_date=date.fromisoformat(search["search_date"]),
        start=date.fromisoformat(search["start"]),
        # 參數的內容由演算法套件決定,這裡以 pickle 原樣保存,不拆開欄位。
        # 代價是 mdrsmap 裡組成參數的類別改名或搬動後,已存的參數會讀不回來
        params=pickle.loads(search["params"]),
        days=days,
    )


def add_daily(
    connection: sqlite3.Connection, stock_id: str, table: pd.DataFrame
) -> None:
    """存下標準每日資料表到最後一個資料齊全的交易日為止;已經存過的交易日保留原本的值。

    最後一個資料齊全的交易日之後的日子不存:那是資料來源還沒公布完的日子,
    下次要重新抓。在它之前的缺值照樣存成 NULL,例如剛上市還沒有籌碼資料的日子。
    """
    complete = table.index[table.notna().all(axis=1)]
    kept = table.loc[: complete[-1]] if len(complete) else table.iloc[:0]
    rows = [
        (stock_id, day.date().isoformat(), *map(_sql_value, values))
        for day, values in zip(
            kept.index, kept.itertuples(index=False, name=None), strict=True
        )
    ]
    with connection:
        connection.executemany(
            f"INSERT OR IGNORE INTO daily (stock_id, date, {_DAILY_COLUMN_LIST})"
            f" VALUES (?, ?, {', '.join('?' * len(DAILY_DTYPES))})",
            rows,
        )


def get_daily(connection: sqlite3.Connection, stock_id: str) -> pd.DataFrame:
    """回傳這支股票存下的每一個交易日,格式是標準每日資料表;還沒存過時是空表。"""
    rows = connection.execute(
        f"SELECT date, {_DAILY_COLUMN_LIST} FROM daily"
        " WHERE stock_id = ? ORDER BY date",
        (stock_id,),
    ).fetchall()
    records = pd.DataFrame(
        [tuple(row) for row in rows], columns=["date", *DAILY_DTYPES]
    )
    table = records[list(DAILY_DTYPES)].astype(DAILY_DTYPES)
    table.index = pd.DatetimeIndex(records["date"], name="date")
    return table


def list_stock_horizons(connection: sqlite3.Connection) -> list[tuple[str, int]]:
    """回傳有逐日預測的每一組股票代號與 Horizon,依股票代號、再依 Horizon 排列。"""
    rows = connection.execute(
        "SELECT DISTINCT stock_id, horizon FROM predictions ORDER BY stock_id, horizon"
    )
    return [(row["stock_id"], row["horizon"]) for row in rows]


def _sql_value(value: object) -> object:
    """把資料表裡的一個值換成 sqlite3 存得進去的值:缺值換成 None。

    numpy 的數字要換成 Python 的 int 或 float:sqlite3 不認得 numpy 的整數,
    會把它當成 bytes 存進去。
    """
    if pd.isna(value):
        return None
    return value.item() if isinstance(value, np.generic) else value


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
