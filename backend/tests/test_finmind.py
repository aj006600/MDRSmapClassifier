"""FinMind 轉接層對 FinMind 回應的處理。

成功的回應轉成標準每日資料表,錯誤的回應轉成資料來源的錯誤。
成功的回應是事先向 FinMind 免費層錄下的(不帶 token),錯誤的回應直接寫在測試裡。
測試不連網路。
"""

import json
from collections.abc import Collection
from datetime import date
from pathlib import Path

import httpx2
import pandas as pd
import pytest

from app.finmind import FinMindSource
from app.market import QuotaExceededError, SourceUnavailableError

RECORDED = Path(__file__).parent / "data" / "finmind"
# 錄下的回應涵蓋的日期區間,每個 dataset 與每支股票都相同
FIRST_DAY = date(2026, 10, 1)
LAST_DAY = date(2026, 10, 14)


def _recorded_finmind(
    requests: list[httpx2.Request], without: Collection[tuple[str, str]] = ()
) -> httpx2.MockTransport:
    """回傳假的 FinMind:只有查詢條件與錄音時相同,才回錄下的回應。

    without 列出要從回應裡拿掉的 (dataset, 日期),模擬 FinMind 沒有那一天的資料。
    """

    def handle(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        params = request.url.params
        dataset = params["dataset"]
        recorded = RECORDED / f"{dataset}_{params['data_id']}.json"
        if (
            params["start_date"] != FIRST_DAY.isoformat()
            or params["end_date"] != LAST_DAY.isoformat()
            or not recorded.exists()
        ):
            return httpx2.Response(404)
        body = json.loads(recorded.read_bytes())
        body["data"] = [
            row for row in body["data"] if (dataset, row["date"]) not in without
        ]
        return httpx2.Response(200, json=body)

    return httpx2.MockTransport(handle)


@pytest.mark.parametrize(
    ("stock_id", "first_row"),
    [
        pytest.param(
            "2330",
            {
                "open": 2490.0,
                "high": 2510.0,
                "low": 2485.0,
                "close": 2510.0,
                "volume": 20666092,
                "trading_value": 51702445610,
                "margin_balance": 30658000,
                "foreign_shares": 17944308863,
                "trust_net_buy": 596450,
                "dealer_net_buy": 602528,
            },
            id="twse",
        ),
        pytest.param(
            "6488",
            {
                "open": 1055.0,
                "high": 1135.0,
                "low": 1045.0,
                "close": 1085.0,
                "volume": 24330441,
                "trading_value": 26580156665,
                "margin_balance": 16584000,
                "foreign_shares": 123918634,
                "trust_net_buy": 983090,
                "dealer_net_buy": 98467,
            },
            id="tpex",
        ),
    ],
)
def test_daily_returns_standard_table(stock_id, first_row):
    source = FinMindSource(transport=_recorded_finmind([]))

    table = source.daily(stock_id, FIRST_DAY, LAST_DAY)

    assert table.dtypes.astype(str).to_dict() == {
        "open": "float64",
        "high": "float64",
        "low": "float64",
        "close": "float64",
        "volume": "int64",
        "trading_value": "int64",
        "margin_balance": "Int64",
        "foreign_shares": "Int64",
        "trust_net_buy": "Int64",
        "dealer_net_buy": "Int64",
    }
    assert isinstance(table.index, pd.DatetimeIndex)
    assert table.index.is_monotonic_increasing
    assert table.index[[0, -1]].tolist() == [
        pd.Timestamp(FIRST_DAY),
        pd.Timestamp(LAST_DAY),
    ]
    # 價格與成交值的單位是元;成交量、融資餘額、外資持股與買賣超的單位是股(不是張)
    assert table.iloc[0].to_dict() == first_row


def test_daily_leaves_chip_data_missing_on_day_not_yet_published():
    # 籌碼資料比價格晚公布。最後一天只有價格時,籌碼欄位是缺值,買賣超也不能當成 0
    last_day = LAST_DAY.isoformat()
    source = FinMindSource(
        transport=_recorded_finmind(
            [],
            without=[
                ("TaiwanStockMarginPurchaseShortSale", last_day),
                ("TaiwanStockShareholding", last_day),
                ("TaiwanStockInstitutionalInvestorsBuySell", last_day),
            ],
        )
    )

    table = source.daily("2330", FIRST_DAY, LAST_DAY)

    chips = ["margin_balance", "foreign_shares", "trust_net_buy", "dealer_net_buy"]
    assert table.index[-1] == pd.Timestamp(LAST_DAY)
    assert table.loc[last_day, chips].isna().all()
    assert table.drop(columns=chips).notna().all().all()


@pytest.mark.parametrize(
    "also_missing",
    [
        pytest.param([], id="chips-published"),
        # 外資持股與融資同時公布;只有融資的日子是 FinMind 漏了外資持股,不是還沒公布
        pytest.param(["TaiwanStockShareholding"], id="foreign-shares-missed"),
    ],
)
def test_daily_counts_day_without_institutional_trades_as_zero_net_buy(also_missing):
    # FinMind 只列出當天有法人交易的股票。冷門股常有好幾天沒有法人交易
    day = "2026-10-06"
    datasets = ["TaiwanStockInstitutionalInvestorsBuySell", *also_missing]
    source = FinMindSource(
        transport=_recorded_finmind(
            [], without=[(dataset, day) for dataset in datasets]
        )
    )

    table = source.daily("2330", FIRST_DAY, LAST_DAY)

    assert table.loc[day, ["trust_net_buy", "dealer_net_buy"]].tolist() == [0, 0]


def test_daily_sends_token_as_bearer():
    requests = []
    FinMindSource(token="my-token", transport=_recorded_finmind(requests)).daily(
        "2330", FIRST_DAY, LAST_DAY
    )
    assert {request.headers["Authorization"] for request in requests} == {
        "Bearer my-token"
    }


@pytest.mark.parametrize("token", [None, ""], ids=["unset", "empty"])
def test_daily_without_token_sends_no_authorization(token):
    requests = []
    FinMindSource(token=token, transport=_recorded_finmind(requests)).daily(
        "2330", FIRST_DAY, LAST_DAY
    )
    assert not any("Authorization" in request.headers for request in requests)


def _finmind_answering(handle) -> FinMindSource:
    """回傳每一個請求都交給 handle 回應的 FinMind 資料來源。"""
    return FinMindSource(transport=httpx2.MockTransport(handle))


def test_daily_raises_quota_exceeded_when_finmind_answers_402():
    # FinMind 文件記載的額度用盡回應
    source = _finmind_answering(
        lambda request: httpx2.Response(
            402,
            json={
                "msg": "Requests reach the upper limit. https://finmindtrade.com/",
                "status": 402,
            },
        )
    )
    with pytest.raises(QuotaExceededError):
        source.daily("2330", FIRST_DAY, LAST_DAY)


def test_daily_raises_source_unavailable_with_finmind_reason_on_error_answer():
    # 內容照抄帶著不合法的 token 向 FinMind 請求時實際收到的回應
    source = _finmind_answering(
        lambda request: httpx2.Response(
            400, json={"msg": "Token is illegal.", "status": 400}
        )
    )
    with pytest.raises(SourceUnavailableError, match="Token is illegal"):
        source.daily("2330", FIRST_DAY, LAST_DAY)


def test_daily_raises_source_unavailable_when_finmind_is_unreachable():
    def refuse(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("Name or service not known", request=request)

    with pytest.raises(SourceUnavailableError):
        _finmind_answering(refuse).daily("2330", FIRST_DAY, LAST_DAY)


def test_daily_raises_source_unavailable_when_finmind_answer_is_unreadable():
    # 例如網路中間的 proxy 回了一頁錯誤頁,狀態碼卻是 200
    source = _finmind_answering(
        lambda request: httpx2.Response(200, text="<html>Bad Gateway</html>")
    )
    with pytest.raises(SourceUnavailableError):
        source.daily("2330", FIRST_DAY, LAST_DAY)


def test_daily_of_unknown_stock_is_empty():
    # FinMind 對不存在的代號回應成功,但沒有任何一天的資料
    table = FinMindSource(transport=_recorded_finmind([])).daily(
        "99999", FIRST_DAY, LAST_DAY
    )
    assert table.empty
