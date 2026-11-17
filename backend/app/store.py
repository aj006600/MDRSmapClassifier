"""SQLite 儲存:工作的狀態,以及每支股票每個 Horizon 的逐日預測。

API 與執行工作的程序各自開連線,共用同一個資料庫檔案。
"""

import sqlite3
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

import pandas as pd

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
]


class Status(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


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


def create_job(connection: sqlite3.Connection, stock_id: str, horizon: int) -> int:
    """新增一個排隊中的工作,回傳它的 id。"""
    with connection:
        cursor = connection.execute(
            "INSERT INTO jobs (stock_id, horizon, status, created_at)"
            " VALUES (?, ?, ?, ?)",
            (stock_id, horizon, Status.QUEUED, _now()),
        )
    return cursor.lastrowid


def get_job(connection: sqlite3.Connection, job_id: int) -> dict | None:
    row = connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    return dict(row) if row else None


def start_job(connection: sqlite3.Connection, job_id: int) -> dict:
    """把工作標為執行中,回傳標記後的工作。"""
    with connection:
        connection.execute(
            "UPDATE jobs SET status = ?, started_at = ? WHERE id = ?",
            (Status.RUNNING, _now(), job_id),
        )
    return get_job(connection, job_id)


def set_stage(connection: sqlite3.Connection, job_id: int, stage: str) -> None:
    with connection:
        connection.execute("UPDATE jobs SET stage = ? WHERE id = ?", (stage, job_id))


def fail_job(connection: sqlite3.Connection, job_id: int, error: str) -> None:
    with connection:
        connection.execute(
            "UPDATE jobs SET status = ?, error = ?, finished_at = ? WHERE id = ?",
            (Status.FAILED, error, _now(), job_id),
        )


def finish_job(connection: sqlite3.Connection, job_id: int, days: pd.DataFrame) -> None:
    """存下工作的逐日預測,並把工作標為成功。

    days 的格式同 RunResult.days。這支股票這個 Horizon 原本存的逐日預測會整個換掉:
    每次執行都重新做 Search,舊的逐日預測是用另一組參數算的。
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
        for day, row in days.iterrows()
    ]
    with connection:
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


def list_stock_horizons(connection: sqlite3.Connection) -> list[tuple[str, int]]:
    """回傳有逐日預測的每一組股票代號與 Horizon,依股票代號、再依 Horizon 排列。"""
    rows = connection.execute(
        "SELECT DISTINCT stock_id, horizon FROM predictions ORDER BY stock_id, horizon"
    )
    return [(row["stock_id"], row["horizon"]) for row in rows]


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
