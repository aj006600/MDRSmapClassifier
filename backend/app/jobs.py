"""在背景程序裡執行工作。

API 發起工作後立刻返回;執行的階段、結果與失敗原因都由背景程序寫進資料庫,
API 再從資料庫讀出來。
"""

import logging
import multiprocessing
from contextlib import closing
from pathlib import Path

from app import store
from app.market import MarketDataSource
from app.runs import RunConfig, run

logger = logging.getLogger(__name__)

# 用 spawn 而不是 fork:API 程序有多條執行緒,fork 一個多執行緒的程序並不安全
_SPAWN = multiprocessing.get_context("spawn")


def launch(
    job_id: int, database: Path, source: MarketDataSource, config: RunConfig
) -> None:
    """在新的程序裡執行一個排隊中的工作,不等它結束。

    背景程序是 daemon:後端結束時會一起被終止,不會拖住後端的關閉。
    """
    _SPAWN.Process(
        target=execute, args=(job_id, database, source, config), daemon=True
    ).start()


def execute(
    job_id: int, database: Path, source: MarketDataSource, config: RunConfig
) -> None:
    """執行一個排隊中的工作,成功時存下結果,失敗時存下原因。"""
    with closing(store.connect(database)) as connection:
        job = store.start_job(connection, job_id)
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
            store.fail_job(connection, job_id, f"{type(error).__name__}: {error}")
        else:
            store.finish_job(connection, job_id, result.days)
