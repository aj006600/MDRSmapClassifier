"""MDR S-map 分類器的兩個對外操作:做 Search 並逐日預測,以及沿用已存的參數逐日預測。

兩個操作都以同一個方式預測每一天:Library 接上那一天的資料,正規化後依參數重新建模。
所以同一天的 Direction 不論是 Search 當下算的、還是之後沿用參數補算的,結果都相同。
"""

from dataclasses import dataclass
from datetime import date

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from mdrsmap import edm
from mdrsmap.config import Config

# Horizon 天前的 Direction。它在當天收盤後已經揭曉,所以能當作當天的特徵
_LAGGED_DIRECTION = "lagged_direction"


@dataclass(frozen=True)
class Params:
    """Search 找到的參數,之後的交易日沿用它預測 Direction。"""

    horizon: int
    start: date
    """Search 時第一個預測的交易日。Library 是特徵表裡這一天之前的所有列。"""
    max_lag: int
    features: tuple[str, ...]
    """CCM 選出的特徵,順序就是模型輸入欄位的順序。"""
    views: tuple[edm.View, ...]
    logistic: edm.Logistic


def search(
    features: pd.DataFrame,
    horizon: int,
    start: date,
    config: Config = Config(),
) -> tuple[Params, pd.Series]:
    """以 start 當天與之前的資料做 Search,再從 start 逐日預測到特徵表的最後一天。

    features 以交易日(DatetimeIndex,由舊到新)為 index,每一欄是一個數值特徵,不能有缺值,
    而且必須有收盤價 `close` 欄。欄名不能含空白,欄位的順序會影響 Search 的結果。
    start 不是交易日時,從它之後的第一個交易日開始。

    回傳找到的參數,以及從 start 起每個交易日的 Direction(0 或 1,以交易日為 index)。
    最後 Horizon 個交易日的實際 Direction 還沒揭曉,它們的預測就是 Forecast。
    """
    table = _modeling_table(features, horizon)
    first = int(table.index.searchsorted(pd.Timestamp(start)))
    data = _library_with(table, first, first)

    target_dimension = edm.target_embedding_dimension(
        data, _LAGGED_DIRECTION, config.validation_days
    )
    selected = edm.select_features(
        data, _LAGGED_DIRECTION, target_dimension, config.max_embedding_dimension
    )
    embedded = edm.embed(data, _LAGGED_DIRECTION, tuple(selected), config.max_lag)
    views = edm.random_views(
        embedded,
        _LAGGED_DIRECTION,
        target_dimension,
        config.validation_days,
        config.random_search_trials,
        config.top_views,
    )
    logistic = edm.select_logistic(
        embedded.filter(like="(t-0)"),
        _LAGGED_DIRECTION,
        edm.weighted_distances(embedded, views),
        config.thetas,
        horizon,
        config.validation_days,
    )
    params = Params(
        horizon=horizon,
        start=table.index[first].date(),
        max_lag=config.max_lag,
        features=tuple(selected),
        views=views,
        logistic=logistic,
    )
    return params, _directions(table, params, first)


def forecast(features: pd.DataFrame, params: Params, since: date) -> pd.Series:
    """沿用 Search 找到的參數,從 since 逐日預測到特徵表的最後一天。

    features 的格式同 search。Library 取自 features 裡 params.start 之前的列,
    這部分要與 Search 時的特徵表相同,結果才會與 Search 當下一次跑完的相同。
    since 不能早於 params.start:更早的日子屬於 Library。

    回傳從 since 起每個交易日的 Direction(0 或 1,以交易日為 index)。
    """
    if pd.Timestamp(since) < pd.Timestamp(params.start):
        raise ValueError(
            f"since ({since}) 早於 Search 的起點 ({params.start}),那些日子屬於 Library"
        )
    table = _modeling_table(features, params.horizon)
    return _directions(
        table, params, int(table.index.searchsorted(pd.Timestamp(since)))
    )


def _modeling_table(features: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """在特徵表最後加上 Horizon 天前的 Direction,並丟掉算不出它的開頭幾列。"""
    close = features["close"]
    lagged = (close.rolling(horizon).mean() - close.shift(horizon) >= 0).astype(float)
    return features.assign(**{_LAGGED_DIRECTION: lagged}).iloc[horizon:]


def _directions(table: pd.DataFrame, params: Params, first: int) -> pd.Series:
    start = int(table.index.searchsorted(pd.Timestamp(params.start)))
    days = range(first, len(table))
    return pd.Series(
        [_predict_day(table, start, day, params) for day in days],
        index=table.index[first:],
        name="direction",
        dtype=int,
    )


def _predict_day(table: pd.DataFrame, start: int, day: int, params: Params) -> int:
    data = _library_with(table, start, day)
    embedded = edm.embed(data, _LAGGED_DIRECTION, params.features, params.max_lag)
    return edm.predict(
        embedded.filter(like="(t-0)"),
        _LAGGED_DIRECTION,
        edm.weighted_distances(embedded, params.views),
        params.logistic,
        params.horizon,
    )


def _library_with(table: pd.DataFrame, start: int, day: int) -> pd.DataFrame:
    """Library(start 之前的所有列)接上第 day 列,正規化成 edm 各步驟吃的格式。"""
    rows = pd.concat([table.iloc[:start], table.iloc[[day]]], ignore_index=True)
    data = pd.DataFrame(MinMaxScaler().fit_transform(rows), columns=rows.columns)
    data.insert(0, edm.TIME, range(1, len(data) + 1))
    return data
