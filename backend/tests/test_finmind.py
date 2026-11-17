"""FinMind 轉接層把 FinMind 的回應轉成標準每日資料表。

回應是事先向 FinMind 免費層錄下的(不帶 token),測試不連網路。
"""

from datetime import date
from pathlib import Path

import httpx2
import pandas as pd
import pytest

from app.finmind import FinMindSource

RECORDED = Path(__file__).parent / "data" / "finmind"
# 錄下的回應涵蓋的日期區間,兩支股票相同
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
