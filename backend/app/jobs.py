"""依發起的順序在背景程序裡執行工作,同一時間只執行一個。

API 發起工作時只把工作記進資料庫就返回。Worker 依序取出排隊中的工作,
每個工作開一個新的程序執行;執行的階段、結果與失敗原因都由那個程序寫進資料庫,
API 再從資料庫讀出來。

這個做法假設後端只有一個程序。多個 uvicorn worker 會各自派工,同一時間就不只跑一個;
每個 worker 啟動時也會把別的 worker 正在跑的工作標成失敗。
"""

import logging
import multiprocessing
import threading
from contextlib import closing
from pathlib import Path

from app import store
from app.market import MarketDataSource, QuotaExceededError, SourceUnavailableError
from app.runs import InsufficientHistoryError, RunConfig, StockNotFoundError, run

logger = logging.getLogger(__name__)

# 用 spawn 而不是 fork:API 程序有多條執行緒,fork 一個多執行緒的程序並不安全
_SPAWN = multiprocessing.get_context("spawn")

_INTERRUPTED_REASON = (
    "服務在這個工作執行途中重新啟動,工作沒有完成。需要結果的話請重新發起。"
)


class Worker:
    """依發起的順序執行排隊中的工作,同一時間只執行一個。

    派工在 API 程序裡的一條執行緒上進行,每個工作在自己的程序裡跑。
    """

    def __init__(
        self, database: Path, source: MarketDataSource, config: RunConfig
    ) -> None:
        self._database = database
        self._source = source
        self._config = config
        self._wake = threading.Event()
        # stop 與派工都要先拿這把鎖:stop 之後就不會再有新的程序被啟動
        self._lock = threading.Lock()
        self._stopping = False
        self._process: multiprocessing.Process | None = None
        self._thread = threading.Thread(target=self._dispatch, daemon=True)

    def start(self) -> None:
        """開始依序執行排隊中的工作,包括上次停止時還在排隊的。

        上次停止時還在執行中的工作不會重跑,而是標為失敗。
        """
        with closing(store.connect(self._database)) as connection:
            store.fail_running_jobs(connection, _INTERRUPTED_REASON)
        self._thread.start()

    def notify(self) -> None:
        """告訴 Worker 有新的工作排進來了。"""
        self._wake.set()

    def stop(self) -> None:
        """停止派工,並終止執行中的工作的程序。

        被終止的工作在資料庫裡仍然是執行中。
        """
        with self._lock:
            self._stopping = True
            if self._process is not None:
                self._process.terminate()
        self._wake.set()
        self._thread.join()

    def _dispatch(self) -> None:
        with closing(store.connect(self._database)) as connection:
            while True:
                # 先清掉通知再找工作:找完之後才排進來的工作,會讓下面的 wait 立刻返回
                self._wake.clear()
                with self._lock:
                    if self._stopping:
                        return
                    job_id = store.claim_next_job(connection)
                    if job_id is not None:
                        self._process = _SPAWN.Process(
                            target=execute,
                            args=(job_id, self._database, self._source, self._config),
                            # daemon 程序會在後端結束時一起被終止,不會拖住後端的關閉
                            daemon=True,
                        )
                        self._process.start()
                if job_id is None:
                    self._wake.wait()
                    continue
                self._process.join()
                # 程序沒有寫下結果就結束時,工作還標為執行中,
                # 例如程序因為記憶體不足被系統終止。
                # stop 終止的程序除外:那個工作留到下次啟動時才標為失敗
                if not self._stopping:
                    exitcode = self._process.exitcode
                    store.fail_running_jobs(
                        connection,
                        f"執行這個工作的程序意外結束(exit code {exitcode}),"
                        "工作沒有完成。",
                    )


def execute(
    job_id: int, database: Path, source: MarketDataSource, config: RunConfig
) -> None:
    """執行一個已經標為執行中的工作,成功時存下結果,失敗時存下原因。"""
    with closing(store.connect(database)) as connection:
        job = store.get_job(connection, job_id)
        try:
            result = run(
                source,
                job["stock_id"],
                job["horizon"],
                config,
                on_stage=lambda stage: store.set_stage(connection, job_id, stage),
            )
        except Exception as error:
            logger.exception("工作 %s 失敗", job_id)
            store.fail_job(connection, job_id, _failure_reason(error))
        else:
            store.finish_job(connection, job_id, result.days)


def _failure_reason(error: Exception) -> str:
    """回傳給使用者看的失敗原因,說明問題出在哪裡、可以怎麼處理。"""
    match error:
        case StockNotFoundError():
            return (
                f"查無股票代號 {error.stock_id}:資料來源沒有這支股票的交易資料,"
                "請確認代號是否正確。"
            )
        case InsufficientHistoryError():
            return (
                f"歷史資料不足:起點之前需要 {error.required} 個交易日的資料當 Library,"
                f"這支股票只有 {error.available} 個交易日。"
            )
        case QuotaExceededError():
            return (
                "資料來源的請求額度用盡了。請稍後再試;"
                "還沒有在 .env 填 FINMIND_TOKEN 的話,填入 token 可以提高額度。"
            )
        case SourceUnavailableError():
            return f"資料來源連不上或回應異常。細節:{error}"
        case _:
            return f"程式錯誤,請回報這個問題。細節:{type(error).__name__}: {error}"
