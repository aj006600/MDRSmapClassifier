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
    "trading_value": "int64",
    "margin_balance": "Int64",
    "foreign_shares": "Int64",
    "trust_net_buy": "Int64",
    "dealer_net_buy": "Int64",
}
"""標準每日資料表的欄位、順序與型別。

各欄的意義與單位:

- open、high、low、close:開盤、最高、最低、收盤價,單位是元
- volume:成交量,單位是股
- trading_value:成交值,單位是元
- margin_balance:當天收盤後的融資餘額,單位是股
- foreign_shares:當天收盤後的外資持股,單位是股
- trust_net_buy、dealer_net_buy:投信與自營商當天的買賣超,單位是股,買超為正

型別是 Int64 的欄位在資料來源沒有那一天的這項資料時是缺值(pd.NA)。
例如籌碼資料比價格晚公布,剛上市的股票還沒有籌碼資料,資料來源也可能漏掉某一天。
資料來源不能自己補值。
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
        欄位、順序、型別與單位見 DAILY_DTYPES,哪些欄位可能缺值也寫在那裡。
        上市與上櫃的股票都用同一個格式。

        資料來源沒有這支股票時回傳空的表。額度用盡時 raise QuotaExceededError;
        連不上或回應異常時 raise SourceUnavailableError。
        """
        ...
