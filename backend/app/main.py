"""後端的 FastAPI 進入點:發起執行、查詢工作,以及查詢某支股票某個 Horizon 的結果。

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


class Backtest(BaseModel):
    accuracy: float | None
    """預測的 Direction 與實際相同的天數比例;沒有計分的日子時是 null。"""
    scored_days: int


class ForecastDay(BaseModel):
    date: dt.date
    """這筆 Forecast 依據的交易日。"""
    direction: int


class Result(BaseModel):
    stock_id: str
    horizon: int
    backtest: Backtest
    forecast: list[ForecastDay]


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


@router.get("/api/results/{stock_id}/{horizon}")
def get_result(stock_id: str, horizon: int, connection: Connection) -> Result:
    """回傳最近一次成功執行的 Backtest 與 Forecast。"""
    days = store.get_predictions(connection, stock_id, horizon)
    if not days:
        raise HTTPException(404, "這支股票的這個 Horizon 還沒有結果")
    scored = [day for day in days if day["actual"] is not None]
    correct = sum(day["direction"] == day["actual"] for day in scored)
    return Result(
        stock_id=stock_id,
        horizon=horizon,
        backtest=Backtest(
            accuracy=correct / len(scored) if scored else None,
            scored_days=len(scored),
        ),
        forecast=[
            ForecastDay(date=day["date"], direction=day["direction"])
            for day in days
            if day["actual"] is None
        ],
    )


configure_logging(settings)
app = create_app(FinMindSource(settings.finmind_token), DATABASE)
