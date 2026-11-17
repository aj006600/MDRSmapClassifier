"""FinMind 轉接層對 FinMind 回應的處理。

成功的回應轉成標準每日資料表,錯誤的回應轉成資料來源的錯誤。
成功的回應是事先向 FinMind 免費層錄下的(不帶 token),錯誤的回應直接寫在測試裡。
測試不連網路。
"""

from datetime import date
from pathlib import Path

import httpx2
import pandas as pd
import pytest

from app.finmind import FinMindSource
from app.market import QuotaExceededError, SourceUnavailableError

RECORDED = Path(__file__).parent / "data" / "finmind"
# 錄下的回應涵蓋的日期區間,每一支股票都相同
FIRST_DAY = date(2026, 10, 1)
LAST_DAY = date(2026, 10, 14)


def _recorded_finmind(requests: list[httpx2.Request]) -> httpx2.MockTransport:
    """回傳假的 FinMind:只有查詢條件與錄音時相同,才回錄下的回應。"""

    def handle(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        params = request.url.params
        recorded = RECORDED / f"TaiwanStockPrice_{params['data_id']}.json"
        if (
            params["dataset"] != "TaiwanStockPrice"
            or params["start_date"] != FIRST_DAY.isoformat()
            or params["end_date"] != LAST_DAY.isoformat()
            or not recorded.exists()
        ):
            return httpx2.Response(404)
        return httpx2.Response(200, content=recorded.read_bytes())

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
    }
    assert isinstance(table.index, pd.DatetimeIndex)
    assert table.index.is_monotonic_increasing
    assert table.index[[0, -1]].tolist() == [
        pd.Timestamp(FIRST_DAY),
        pd.Timestamp(LAST_DAY),
    ]
    # 價格的單位是元,成交量的單位是股(不是張)
    assert table.iloc[0].to_dict() == first_row


def test_daily_sends_token_as_bearer():
    requests = []
    FinMindSource(token="my-token", transport=_recorded_finmind(requests)).daily(
        "2330", FIRST_DAY, LAST_DAY
    )
    assert requests[0].headers["Authorization"] == "Bearer my-token"


@pytest.mark.parametrize("token", [None, ""], ids=["unset", "empty"])
def test_daily_without_token_sends_no_authorization(token):
    requests = []
    FinMindSource(token=token, transport=_recorded_finmind(requests)).daily(
        "2330", FIRST_DAY, LAST_DAY
    )
    assert "Authorization" not in requests[0].headers


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
