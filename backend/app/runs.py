"""一次執行:對一支股票與一個 Horizon 抓資料、做 Search,再逐日預測到最新的交易日。

最新交易日是最近一個資料齊全的交易日:資料來源還沒公布完的交易日不算進來。
起點是最新交易日往前 lookback_days 個交易日。Search 只拿到起點當天收盤時已知的資料:
起點之前的 Library,加上起點當天那一列。起點之後的每一天沿用 Search 找到的參數預測。
已經知道實際 Direction 的日子屬於 Backtest;最後 Horizon 個交易日還沒揭曉,
它們的預測就是 Forecast。
"""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, timedelta
from enum import StrEnum

import pandas as pd

import mdrsmap
from app.features import feature_table
from app.market import MarketDataSource


class StockNotFoundError(Exception):
    """資料來源在抓資料的區間裡沒有這支股票的任何一個交易日。"""

    def __init__(self, stock_id: str) -> None:
        super().__init__(stock_id)
        self.stock_id = stock_id


class InsufficientHistoryError(Exception):
    """起點之前的交易日比設定的 Library 少。"""

    def __init__(self, required: int, available: int) -> None:
        super().__init__(required, available)
        self.required = required
        self.available = available


class Stage(StrEnum):
    """執行中回報的階段,依宣告的順序發生。"""

    FETCH = "fetch"
    SEARCH = "search"
    PREDICT = "predict"


@dataclass(frozen=True)
class RunConfig:
    """一次執行的設定值,預設為服務使用的值。"""

    lookback_days: int = 100
    """起點在最新交易日往前幾個交易日。"""

    library_days: int = 600
    """Library 取起點之前幾個交易日,也是執行一支股票的最低門檻。

    600 個交易日約兩年半,與論文的 Library 相當。
    """

    algorithm: mdrsmap.Config = mdrsmap.Config()


@dataclass(frozen=True)
class RunResult:
    start: date
    params: mdrsmap.Params
    days: pd.DataFrame
    """從起點到最新交易日的每一天,以交易日為 index。

    direction 欄是預測的 Direction。actual 欄是實際的 Direction,還沒揭曉的日子是 pd.NA。
    """


def run(
    source: MarketDataSource,
    stock_id: str,
    horizon: int,
    config: RunConfig = RunConfig(),
    on_stage: Callable[[Stage], None] = lambda stage: None,
) -> RunResult:
    """執行一次,每進入一個階段就以那個階段呼叫 on_stage。

    資料來源沒有這支股票的任何交易日時 raise StockNotFoundError;
    起點之前不足 library_days 個交易日時 raise InsufficientHistoryError。
    """
    on_stage(Stage.FETCH)
    trading_days = config.library_days + config.lookback_days + 1
    today = date.today()
    daily = source.daily(stock_id, _fetch_from(today, trading_days), today)
    if daily.empty:
        raise StockNotFoundError(stock_id)
    table = feature_table(daily).iloc[-trading_days:]
    if len(table) < trading_days:
        raise InsufficientHistoryError(
            config.library_days, max(len(table) - config.lookback_days - 1, 0)
        )
    start = table.index[-1 - config.lookback_days]

    on_stage(Stage.SEARCH)
    params, start_day = mdrsmap.search(
        table.loc[:start], horizon, start, config.algorithm
    )

    on_stage(Stage.PREDICT)
    after_start = mdrsmap.forecast(table, params, since=start + pd.Timedelta(days=1))
    direction = pd.concat([start_day, after_start])
    actual = _actual_directions(table["close"], horizon)
    return RunResult(
        start=start.date(),
        params=params,
        days=pd.DataFrame({"direction": direction, "actual": actual[direction.index]}),
    )


def _fetch_from(today: date, trading_days: int) -> date:
    """回傳抓資料的起始日,要涵蓋 trading_days 個交易日與技術指標開頭算不出來的日子。

    技術指標開頭算不出來的日子不到 60 個交易日。一年約 245 個交易日,
    換算成日曆天約乘 1.5;這裡乘 2,留給長假的餘裕。
    """
    return today - timedelta(days=2 * (trading_days + 60))


def _actual_directions(close: pd.Series, horizon: int) -> pd.Series:
    """回傳每一天的實際 Direction;最後 horizon 個交易日還沒揭曉,是 pd.NA。"""
    future_mean = close.rolling(horizon).mean().shift(-horizon)
    return (future_mean >= close).astype("Int64").where(future_mean.notna())
