"""論文的 MDR S-map 分類器:給一支股票的每日特徵表,逐日預測 Direction。

這個套件獨立於網頁 API:它不知道 HTTP、資料庫與資料來源,不寫任何檔案,也不印出任何文字。
對外有兩個操作與一個設定物件:

- search:做 Search,並從起點逐日預測到特徵表的最後一天
- forecast:沿用 search 回傳的 Params,從指定的日子逐日預測到特徵表的最後一天
- Config:演算法的設定值,預設為論文的值

Params 是 search 回傳、forecast 接收的參數,呼叫端只負責保存它,不必自己建立。
"""

from mdrsmap.classifier import Params, forecast, search
from mdrsmap.config import Config

__all__ = ["Config", "Params", "forecast", "search"]
