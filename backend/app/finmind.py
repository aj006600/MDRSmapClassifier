"""以 FinMind 免費層實作的市場資料來源。

FinMind 的 TaiwanStockPrice dataset 同時收錄上市與上櫃的股票,格式相同,
所以這一層不必知道股票掛在哪個市場。
"""

from dataclasses import dataclass
from datetime import date

import httpx2
import pandas as pd

from app.market import DAILY_DTYPES, QuotaExceededError, SourceUnavailableError

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
        try:
            with httpx2.Client(transport=self.transport, timeout=60) as client:
                response = client.get(URL, params=params, headers=headers)
        except httpx2.HTTPError as error:
            raise SourceUnavailableError(
                f"連不上 FinMind,請稍後再試({error})"
            ) from error
        # FinMind 文件記載:請求額度用盡時回 402
        if response.status_code == 402:
            raise QuotaExceededError()
        if response.status_code != 200:
            raise SourceUnavailableError(
                f"FinMind 回應 HTTP {response.status_code}({_reason(response)})"
            )

        try:
            data = response.json()["data"]
        except (ValueError, KeyError, TypeError) as error:
            raise SourceUnavailableError("FinMind 回應的內容讀不出資料") from error
        records = pd.DataFrame(data, columns=["date", *_COLUMNS])
        table = records[list(_COLUMNS)].rename(columns=_COLUMNS).astype(DAILY_DTYPES)
        table.index = pd.DatetimeIndex(records["date"], name="date")
        return table.sort_index()


def _reason(response: httpx2.Response) -> str:
    """回傳 FinMind 在錯誤回應裡附的說明。

    回應不是 FinMind 的格式時(例如中間的 proxy 回的錯誤頁),回傳 HTTP 的狀態說明。
    """
    try:
        return response.json()["msg"]
    except (ValueError, KeyError, TypeError):
        return response.reason_phrase
