"""市場資料來源的介面。

執行流程只透過這個介面取得每日資料,不知道資料實際從哪裡來;換資料來源時只要換掉實作。
"""

from datetime import date
from typing import Protocol

import pandas as pd

DAILY_DTYPES = {
    "open": "float64",
    "high": "float64",
    "low": "float64",
    "close": "float64",
    "volume": "int64",
}
"""標準每日資料表的欄位、順序與型別。

價格(開盤、最高、最低、收盤)的單位是元,成交量的單位是股。
"""


class QuotaExceededError(Exception):
    """資料來源的請求額度用盡了。"""


class SourceUnavailableError(Exception):
    """資料來源連不上,或回應的內容不能用。

    訊息是給使用者看的細節,說明是哪個資料來源、發生了什麼事。
    """


class MarketDataSource(Protocol):
    """給定股票代號與日期區間,回傳標準每日資料表的資料來源。

    實作必須能被 pickle:執行在另一個程序裡呼叫它。
    """

    def daily(self, stock_id: str, start: date, end: date) -> pd.DataFrame:
        """回傳 start 到 end(都含)之間每個交易日的標準每日資料表。

        標準每日資料表的 index 是交易日(DatetimeIndex),由舊到新排列;
        欄位、順序與型別見 DAILY_DTYPES。上市與上櫃的股票都用同一個格式。
        某個交易日的資料還沒公布完時,還沒公布的欄位是缺值。

        資料來源沒有這支股票時回傳空的表。額度用盡時 raise QuotaExceededError;
        連不上或回應異常時 raise SourceUnavailableError。
        """
        ...
