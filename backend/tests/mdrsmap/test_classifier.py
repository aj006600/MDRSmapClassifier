"""search 與 forecast 的對外行為。

回歸測試以論文的 2330 資料與設定執行,驗證重構沒有改變演算法算出的結果。
標準答案 data/2330_directions.csv 在鎖定的環境(backend 的 dev stage 容器)下
以這個套件產生,產生時已確認逐日預測與搬進套件之前的論文程式完全相同。
只有刻意改變演算法時才重新產生它。
"""

import multiprocessing
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from mdrsmap import Config, forecast, search

PAPER_HORIZON = 10
PAPER_START = date(2023, 7, 1)
PAPER_BACKTEST_END = "2023-10-31"
GOLDEN = Path(__file__).parent / "data" / "2330_directions.csv"

# 調小到 Search 幾秒內跑完。結果不對應論文,只用來驗證操作之間的一致性
SMALL = Config(
    top_views=5,
    max_embedding_dimension=2,
    max_lag=3,
    thetas=(1, 4),
    validation_days=20,
    random_search_trials=20,
)


def _actual_direction(close: pd.Series, horizon: int) -> pd.Series:
    """詞彙表的 Direction:未來 horizon 個交易日的平均收盤價不低於當天的收盤價。"""
    future_mean = close.rolling(horizon).mean().shift(-horizon)
    return (future_mean >= close).astype(int)


@pytest.fixture(scope="module")
def paper_search(paper_features):
    return search(paper_features, PAPER_HORIZON, PAPER_START)


@pytest.fixture(scope="module")
def small_features(paper_features):
    return paper_features.loc[
        "2022-07-01":"2023-07-31", ["close", "volume", "MACD", "RSI14"]
    ]


@pytest.fixture(scope="module")
def small_search(small_features):
    return search(small_features, PAPER_HORIZON, PAPER_START, SMALL)


@pytest.mark.slow
def test_search_keeps_backtest_accuracy_on_paper_data(paper_features, paper_search):
    _, directions = paper_search
    scored = directions.loc[:PAPER_BACKTEST_END]
    actual = _actual_direction(paper_features["close"], PAPER_HORIZON).loc[scored.index]
    # 搬進套件之前的論文程式在同一個環境下算出的準確率:83 天裡對 46 天
    assert (scored == actual).mean() == 0.5542168674698795


@pytest.mark.slow
def test_search_directions_match_golden(paper_search):
    _, directions = paper_search
    golden = pd.read_csv(GOLDEN, dtype={"date": str})
    assert dict(
        zip(directions.index.strftime("%Y-%m-%d"), directions, strict=True)
    ) == dict(zip(golden["date"], golden["direction"], strict=True))


def test_forecast_with_saved_params_matches_one_search_run(
    small_features, small_search
):
    params, searched = search(
        small_features.loc[:"2023-07-14"], PAPER_HORIZON, PAPER_START, SMALL
    )
    continued = forecast(
        small_features, params, since=searched.index[-1] + pd.Timedelta(days=1)
    )
    pd.testing.assert_series_equal(pd.concat([searched, continued]), small_search[1])


def test_forecast_rejects_days_inside_library(small_features, small_search):
    params, _ = small_search
    with pytest.raises(ValueError, match="Library"):
        forecast(small_features, params, since=date(2023, 6, 30))


def test_search_in_spawned_process_matches_in_process(small_features, small_search):
    # 執行流程會在另一個程序裡呼叫演算法。spawn 會在新的直譯器裡重新 import 呼叫端,
    # 而且輸入與結果都要能 pickle
    with multiprocessing.get_context("spawn").Pool(1) as pool:
        _, directions = pool.apply(
            search, (small_features, PAPER_HORIZON, PAPER_START, SMALL)
        )
    pd.testing.assert_series_equal(directions, small_search[1])


def test_search_prints_nothing(small_features, capfd):
    search(small_features.loc[:"2023-07-07"], PAPER_HORIZON, PAPER_START, SMALL)
    assert capfd.readouterr().out == ""
