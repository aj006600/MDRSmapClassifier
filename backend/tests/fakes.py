"""測試用的假資料來源。

這些類別放在獨立的模組,不放在 conftest:執行在另一個程序裡呼叫資料來源,
子程序要能以模組名稱重新 import 這些類別,才能 unpickle 資料來源。
"""

from dataclasses import dataclass
from datetime import date

import pandas as pd


@dataclass
class FixedSource:
    """不論股票代號,都回傳固定的每日資料表裡落在查詢區間的列。"""

    table: pd.DataFrame

    def daily(self, stock_id: str, start: date, end: date) -> pd.DataFrame:
        return self.table.loc[pd.Timestamp(start) : pd.Timestamp(end)]


@dataclass
class FailingSource:
    """每次查詢都失敗的資料來源。"""

    message: str

    def daily(self, stock_id: str, start: date, end: date) -> pd.DataFrame:
        raise ConnectionError(self.message)
