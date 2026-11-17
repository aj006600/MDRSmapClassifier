"""後端的 FastAPI 進入點,提供執行、工作與結果的 HTTP API。

端點涵蓋發起執行、查詢工作、列出有結果的股票,以及查詢某支股票某個 Horizon 的結果。

create_app 接收資料來源、資料庫檔案與執行的設定;模組層的 app 是 uvicorn 載入的正式設定。
"""

import datetime as dt
import sqlite3
from collections.abc import Iterator
from contextlib import asynccontextmanager, closing
from pathlib import Path
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
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
    data_date: dt.date
    """執行時的資料用到的最後一個交易日。"""
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
    source: MarketDataSource, database: Path, config: RunConfig = RunConfig()
) -> FastAPI:
    """建立以 source 取得資料、把狀態與結果存進 database 的 app。

    啟動時會把 database 的結構更新到最新。
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        with closing(store.connect(database)) as connection:
            store.migrate(connection)
        yield

    app = FastAPI(
        title="mdrsmap-classifier backend", version="0.1.0", lifespan=lifespan
    )
    app.state.source = source
    app.state.database = database
    app.state.config = config
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
def start_job(body: JobRequest, connection: Connection, request: Request) -> Job:
    """發起一次執行。執行在背景跑,以回傳的 id 查詢進度。"""
    job_id = store.create_job(connection, body.stock_id, body.horizon)
    job = Job(**store.get_job(connection, job_id))
    state = request.app.state
    jobs.launch(job_id, state.database, state.source, state.config)
    return job


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
        days = store.get_predictions(connection, stock_id, horizon)
        result = _build_result(stock_id, horizon, days)
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
    """回傳最近一次成功執行的 Backtest 與 Forecast。"""
    days = store.get_predictions(connection, stock_id, horizon)
    if not days:
        raise HTTPException(404, "這支股票的這個 Horizon 還沒有結果")
    return _build_result(stock_id, horizon, days)


def _build_result(stock_id: str, horizon: int, days: list[dict]) -> Result:
    """從不是空的逐日預測算出結果;days 的格式同 store.get_predictions。

    最後 Horizon 個交易日的實際 Direction 一定還沒揭曉,所以 Forecast 不會是空的。
    """
    scored = [day for day in days if day["actual"] is not None]
    correct = sum(day["direction"] == day["actual"] for day in scored)
    ones = sum(day["actual"] == 1 for day in scored)
    return Result(
        stock_id=stock_id,
        horizon=horizon,
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
