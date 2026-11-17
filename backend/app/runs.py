"""一次執行:對一支股票與一個 Horizon 逐日預測到最新的交易日。

最新交易日是最近一個資料齊全的交易日:資料來源還沒公布完的交易日不算進來。
第一次執行時做 Search,起點是最新交易日往前 lookback_days 個交易日。
Search 只拿到起點當天收盤時已知的資料:起點之前的 Library,加上起點當天那一列。
起點之後的每一天沿用 Search 找到的參數預測。之後的執行沿用同一組參數與起點,
只預測新增的交易日。

已經知道實際 Direction 的日子屬於 Backtest;最後 Horizon 個交易日還沒揭曉,
它們的預測就是 Forecast。Forecast 的實際 Direction 揭曉後,下一次執行把它轉入 Backtest。
"""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, timedelta
from enum import StrEnum

import pandas as pd

import mdrsmap
from app.features import feature_table

# 特徵表開頭算不出技術指標的日子,以及最新幾天還沒公布籌碼資料的日子,
# 加起來不到這麼多個交易日
_WARMUP_DAYS = 60


class StockNotFoundError(Exception):
    """這支股票連一個交易日的資料都沒有:資料來源在抓資料的區間裡查不到它。"""

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
    """執行中回報的階段,依宣告的順序發生;沿用參數時沒有 Search 階段。"""

    FETCH = "fetch"
    SEARCH = "search"
    PREDICT = "predict"


@dataclass(frozen=True)
class RunConfig:
    """一次執行的設定值,預設為服務使用的值。"""

    lookback_days: int = 100
    """做 Search 時,起點設在最新交易日往前幾個交易日。"""

    library_days: int = 600
    """Library 取起點之前幾個交易日,也是執行一支股票的最低門檻。

    600 個交易日約兩年半,與論文的 Library 相當。
    """

    algorithm: mdrsmap.Config = mdrsmap.Config()


@dataclass(frozen=True)
class RunResult:
    search_date: date
    """做 Search 的那一天,也就是當時那次執行以哪一天為準。"""
    start: date
    """起點:Search 只用這一天(含)以前的資料,逐日預測從這一天開始。"""
    params: mdrsmap.Params
    days: pd.DataFrame
    """從起點到最新交易日的每一天,以交易日為 index。

    direction 欄是預測的 Direction。actual 欄是實際的 Direction,還沒揭曉的日子是 pd.NA。
    """


def run(
    daily: pd.DataFrame,
    horizon: int,
    as_of: date,
    previous: RunResult | None = None,
    config: RunConfig = RunConfig(),
    on_stage: Callable[[Stage], None] = lambda stage: None,
) -> RunResult:
    """從標準每日資料表 daily 逐日預測到它的最後一天,回傳這次的結果。

    daily 的最後一天要是資料齊全的交易日。沒有 previous 時做 Search,
    Search 的日期記為 as_of。有 previous 時沿用它的參數與起點,
    只預測 previous 最後一天之後的交易日,不做 Search。
    進入 Search 與逐日預測階段時,以那個階段呼叫 on_stage。

    做 Search 時,起點之前不足 library_days 個交易日就 raise InsufficientHistoryError。
    """
    if previous is not None:
        start = pd.Timestamp(previous.start)
    elif len(daily) > config.lookback_days:
        start = daily.index[-1 - config.lookback_days]
    else:
        raise InsufficientHistoryError(config.library_days, 0)
    # 特徵從起點往前固定的交易日數算起。技術指標的值受資料從哪一天開始影響,
    # 同一個起點每次算出的特徵才會相同,沿用參數補算的結果也才會與一次跑完的相同。
    # 更早的日子用不到,有缺漏時也不會讓執行失敗
    first = max(0, daily.index.get_loc(start) - config.library_days - _WARMUP_DAYS)
    features = feature_table(daily.iloc[first:])

    library_length = int((features.index < start).sum())
    if previous is None and library_length < config.library_days:
        raise InsufficientHistoryError(config.library_days, library_length)
    table = features.iloc[max(0, library_length - config.library_days) :]

    if previous is None:
        on_stage(Stage.SEARCH)
        params, predicted = mdrsmap.search(
            table.loc[:start], horizon, start, config.algorithm
        )
        search_date = as_of
    else:
        params = previous.params
        predicted = previous.days["direction"]
        search_date = previous.search_date

    on_stage(Stage.PREDICT)
    since = predicted.index[-1] + pd.Timedelta(days=1)
    direction = pd.concat([predicted, mdrsmap.forecast(table, params, since)])
    actual = _actual_directions(table["close"], horizon)
    return RunResult(
        search_date=search_date,
        start=start.date(),
        params=params,
        days=pd.DataFrame({"direction": direction, "actual": actual[direction.index]}),
    )


def fetch_from(as_of: date, config: RunConfig) -> date:
    """回傳第一次抓一支股票的資料時,要從哪一天抓起。

    抓的範圍要涵蓋以 as_of 為準做一次 Search 需要的交易日:Library、起點到最新的
    交易日,以及特徵表開頭算不出來的日子。一年約 245 個交易日,換算成日曆天約乘 1.5;
    這裡乘 2,留給長假的餘裕。
    """
    trading_days = config.library_days + config.lookback_days + 1
    return as_of - timedelta(days=2 * (trading_days + _WARMUP_DAYS))


def _actual_directions(close: pd.Series, horizon: int) -> pd.Series:
    """回傳每一天的實際 Direction;最後 horizon 個交易日還沒揭曉,是 pd.NA。"""
    future_mean = close.rolling(horizon).mean().shift(-horizon)
    return (future_mean >= close).astype("Int64").where(future_mean.notna())
