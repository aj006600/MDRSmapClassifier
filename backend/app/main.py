"""後端的 FastAPI 進入點,提供執行、工作與結果的 HTTP API。

端點涵蓋發起執行、查詢工作、列出進行中的工作、列出有結果的股票,
以及查詢某支股票某個 Horizon 的結果。

create_app 接收資料來源、資料庫檔案與執行的設定;模組層的 app 是 uvicorn 載入的正式設定。
"""

import datetime as dt
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import asynccontextmanager, closing
from pathlib import Path
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, Response
from pydantic import BaseModel

from app import jobs, store
from app.config import settings
from app.finmind import FinMindSource
from app.logging_setup import configure_logging
from app.market import MarketDataSource
from app.runs import RunConfig, Stage
from app.store import Status

DATABASE = Path("/data/mdrsmap.db")
"""正式的資料庫檔案。/data 是 compose 掛上的持久化 volume,服務重啟後資料還在。"""


class JobRequest(BaseModel):
    stock_id: str
    horizon: Literal[5, 10, 20] = 10
    re_search: bool = False
    """為 true 時不沿用已存的參數:重設起點並重新做 Search,之前的逐日預測全部換掉。"""


class Job(BaseModel):
    id: int
    stock_id: str
    horizon: int
    status: Status
    stage: Stage | None
    """執行中目前的階段;工作結束後保留最後進入的階段。"""
    error: str | None
    created_at: dt.datetime
    started_at: dt.datetime | None
    finished_at: dt.datetime | None


class BacktestDay(BaseModel):
    date: dt.date
    direction: int
    """這一天預測的 Direction。"""
    actual: int
    """這一天實際的 Direction。"""


class BacktestSummary(BaseModel):
    accuracy: float | None
    """預測的 Direction 與實際相同的天數比例;沒有計分的日子時是 null。"""
    scored_days: int
    direction_one_ratio: float | None
    """計分的日子裡實際 Direction 為 1 的天數比例,等於每天都猜 1 的準確率。

    沒有計分的日子時是 null。
    """


class Backtest(BacktestSummary):
    days: list[BacktestDay]
    """計分的每一天,由舊到新排列。"""


class ForecastDay(BaseModel):
    date: dt.date
    """這筆 Forecast 依據的交易日。"""
    direction: int


class Result(BaseModel):
    stock_id: str
    horizon: int
    search_date: dt.date
    """做 Search 找到這組參數的那一天。"""
    start: dt.date
    """起點:逐日預測從這一天開始;Search 只用這一天(含)以前的資料。"""
    data_date: dt.date
    """執行時的資料用到的最後一個交易日。

    最新交易日的資料還沒公布完時,是在它之前最近一個資料齊全的交易日。
    """
    backtest: Backtest
    forecast: list[ForecastDay]


class ResultSummary(BaseModel):
    """一支股票一個 Horizon 最近一次成功執行的摘要。"""

    stock_id: str
    horizon: int
    data_date: dt.date
    """執行時的資料用到的最後一個交易日。"""
    latest_forecast: ForecastDay
    """資料用到的最後一個交易日的 Forecast。"""
    backtest: BacktestSummary


def create_app(
    source: MarketDataSource,
    database: Path,
    config: RunConfig = RunConfig(),
    today: Callable[[], dt.date] = dt.date.today,
) -> FastAPI:
    """建立以 source 取得資料、把狀態與結果存進 database 的 app。

    啟動時會把 database 的結構更新到最新。today 回傳今天的日期,
    只給測試換成指定的日子,正式使用時保持預設。
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        with closing(store.connect(database)) as connection:
            store.migrate(connection)
        app.state.worker = jobs.Worker(database, source, config)
        app.state.worker.start()
        yield
        app.state.worker.stop()

    app = FastAPI(
        title="mdrsmap-classifier backend", version="0.1.0", lifespan=lifespan
    )
    app.state.database = database
    app.state.today = today
    app.include_router(router)
    return app


def _open_connection(request: Request) -> Iterator[sqlite3.Connection]:
    with closing(store.connect(request.app.state.database)) as connection:
        yield connection


Connection = Annotated[sqlite3.Connection, Depends(_open_connection)]

router = APIRouter()


@router.get("/health")
def health_check():
    return {"status": "ok"}


@router.post("/api/jobs", status_code=202)
def start_job(
    body: JobRequest, connection: Connection, request: Request, response: Response
) -> Job:
    """發起一次以今天為準的執行。執行在背景依發起的順序一次跑一個,以回傳的 id 查詢進度。

    沒有要求重新搜尋、且今天已經有成功的工作時,不重算也不抓資料,
    以 HTTP 200 回傳那個工作。
    """
    today = request.app.state.today()
    if not body.re_search:
        done = store.find_succeeded_job(connection, body.stock_id, body.horizon, today)
        if done is not None:
            response.status_code = 200
            return Job(**done)
    job_id = store.create_job(
        connection, body.stock_id, body.horizon, today, body.re_search
    )
    job = Job(**store.get_job(connection, job_id))
    request.app.state.worker.notify()
    return job


@router.get("/api/jobs")
def list_jobs(connection: Connection) -> list[Job]:
    """列出排隊中與執行中的工作,依執行的順序排列。"""
    return [Job(**job) for job in store.list_active_jobs(connection)]


@router.get("/api/jobs/{job_id}")
def get_job(job_id: int, connection: Connection) -> Job:
    job = store.get_job(connection, job_id)
    if job is None:
        raise HTTPException(404, "找不到這個工作")
    return Job(**job)


@router.get("/api/results")
def list_results(connection: Connection) -> list[ResultSummary]:
    """列出每一組有結果的股票與 Horizon,依股票代號、再依 Horizon 排列。"""
    summaries = []
    for stock_id, horizon in store.list_stock_horizons(connection):
        result = _get_result(connection, stock_id, horizon)
        summaries.append(
            ResultSummary(
                stock_id=stock_id,
                horizon=horizon,
                data_date=result.data_date,
                latest_forecast=result.forecast[-1],
                backtest=BacktestSummary(
                    **result.backtest.model_dump(exclude={"days"})
                ),
            )
        )
    return summaries


@router.get("/api/results/{stock_id}/{horizon}")
def get_result(stock_id: str, horizon: int, connection: Connection) -> Result:
    """回傳最近一次成功執行的結果:Search 的日期與起點、Backtest 與 Forecast。"""
    result = _get_result(connection, stock_id, horizon)
    if result is None:
        raise HTTPException(404, "這支股票的這個 Horizon 還沒有結果")
    return result


def _get_result(
    connection: sqlite3.Connection, stock_id: str, horizon: int
) -> Result | None:
    """從存下的 Search 與逐日預測算出結果;還沒有結果時是 None。

    逐日預測從起點開始,所以第一天就是起點。最後 Horizon 個交易日的實際 Direction
    一定還沒揭曉,所以 Forecast 不會是空的。
    """
    search_date = store.get_search_date(connection, stock_id, horizon)
    if search_date is None:
        return None
    days = store.get_predictions(connection, stock_id, horizon)
    scored = [day for day in days if day["actual"] is not None]
    correct = sum(day["direction"] == day["actual"] for day in scored)
    ones = sum(day["actual"] == 1 for day in scored)
    return Result(
        stock_id=stock_id,
        horizon=horizon,
        search_date=search_date,
        start=days[0]["date"],
        data_date=days[-1]["date"],
        backtest=Backtest(
            accuracy=correct / len(scored) if scored else None,
            scored_days=len(scored),
            direction_one_ratio=ones / len(scored) if scored else None,
            days=[BacktestDay(**day) for day in scored],
        ),
        forecast=[
            ForecastDay(date=day["date"], direction=day["direction"])
            for day in days
            if day["actual"] is None
        ],
    )


configure_logging(settings)
app = create_app(FinMindSource(settings.finmind_token), DATABASE)
