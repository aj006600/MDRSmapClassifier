"""特徵表的對外行為。"""

import pandas as pd
import pytest

from app.features import IncompleteDataError, feature_table


def test_feature_table_uses_no_data_after_each_day(daily):
    # 任何一個特徵用到之後的資料,只給前段算出來的值就會和給全部算出來的不同
    last_day = daily.index[199]
    pd.testing.assert_frame_equal(
        feature_table(daily.loc[:last_day]),
        feature_table(daily).loc[:last_day],
        check_exact=True,
    )


def test_feature_table_lists_features_in_order(daily):
    # 欄位順序會影響 Search 的結果
    assert feature_table(daily).columns.tolist() == [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "margin_change",
        "foreign_change",
        "trust_net_buy",
        "dealer_net_buy",
        "EMA9",
        "EMA12",
        "EMA26",
        "MACD",
        "Signal",
        "RSI14",
    ]


def test_feature_table_takes_daily_change_of_balances_and_net_buys_as_is(daily):
    # 最後兩天換成手算得出結果的值
    changed = daily.copy()
    changed.loc[changed.index[-2:], ["margin_balance", "foreign_shares"]] = [
        [5_000_000, 900_000_000],
        [4_998_000, 900_350_000],
    ]
    last_day = changed.index[-1]
    changed.loc[last_day, ["trust_net_buy", "dealer_net_buy"]] = [-12_000, 3_000]

    features = feature_table(changed)

    chips = features.loc[
        last_day, ["margin_change", "foreign_change", "trust_net_buy", "dealer_net_buy"]
    ]
    assert chips.tolist() == [-2_000, 350_000, -12_000, 3_000]


def test_feature_table_starts_at_first_day_with_complete_data(daily):
    # 剛上市的股票要過一段時間才有融資資料,就像那之前還沒上市
    changed = daily.copy()
    changed.loc[changed.index[:100], "margin_balance"] = pd.NA

    pd.testing.assert_frame_equal(
        feature_table(changed), feature_table(daily.iloc[100:])
    )


def test_feature_table_ends_at_latest_day_with_complete_data(daily):
    # 籌碼資料比價格晚公布,最新的幾天可能只有價格
    changed = daily.copy()
    changed.loc[changed.index[-2:], "margin_balance"] = pd.NA

    assert feature_table(changed).index[-1] == daily.index[-3]


@pytest.mark.parametrize(
    ("balance", "change"),
    [("margin_balance", "margin_change"), ("foreign_shares", "foreign_change")],
)
def test_feature_table_carries_balance_over_single_missing_day(daily, balance, change):
    # 資料來源偶爾漏掉某一天的餘額。缺的那天沿用前一天,隔天的日差就包含兩天的變化
    changed = daily.copy()
    days = changed.index[149:152]
    changed.loc[days, balance] = [5_000_000, pd.NA, 5_006_000]

    features = feature_table(changed)

    assert features.loc[days[1:], change].tolist() == [0, 6_000]


@pytest.mark.parametrize(
    "missing",
    [
        pytest.param({"foreign_shares": 2}, id="balance-two-days"),
        # 買賣超是當天的流量,沒有前一天的值可以沿用
        pytest.param({"trust_net_buy": 1}, id="net-buy-one-day"),
    ],
)
def test_feature_table_rejects_gap_between_complete_days(daily, missing):
    changed = daily.copy()
    gap = changed.index[150]
    for column, length in missing.items():
        changed.loc[changed.index[150 : 150 + length], column] = pd.NA

    with pytest.raises(IncompleteDataError, match=gap.date().isoformat()):
        feature_table(changed)
