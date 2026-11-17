"""以 FinMind 免費層實作的市場資料來源。

FinMind 的 TaiwanStockPrice dataset 同時收錄上市與上櫃的股票,格式相同,
所以這一層不必知道股票掛在哪個市場。
"""

from dataclasses import dataclass
from datetime import date

import httpx2
import pandas as pd

from app.market import DAILY_DTYPES

URL = "https://api.finmindtrade.com/api/v4/data"

# FinMind 的欄名對到標準每日資料表的欄名。Trading_Volume 的單位本來就是股
_COLUMNS = {
    "open": "open",
    "max": "high",
    "min": "low",
    "close": "close",
    "Trading_Volume": "volume",
}


@dataclass(frozen=True)
class FinMindSource:
    """從 FinMind 的 TaiwanStockPrice dataset 取得每日資料。

    token 是選配;沒有 token(或是空字串)時,用 FinMind 給未註冊使用者的額度。
    transport 只給測試換成假的 HTTP 回應,正式使用時保持 None。
    """

    token: str | None = None
    transport: httpx2.BaseTransport | None = None

    def daily(self, stock_id: str, start: date, end: date) -> pd.DataFrame:
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        params = {
            "dataset": "TaiwanStockPrice",
            "data_id": stock_id,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
        }
        with httpx2.Client(transport=self.transport, timeout=60) as client:
            response = client.get(URL, params=params, headers=headers)
        response.raise_for_status()

        records = pd.DataFrame(response.json()["data"], columns=["date", *_COLUMNS])
        table = records[list(_COLUMNS)].rename(columns=_COLUMNS).astype(DAILY_DTYPES)
        table.index = pd.DatetimeIndex(records["date"], name="date")
        return table.sort_index()
