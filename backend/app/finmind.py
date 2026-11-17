"""以 FinMind 免費層實作的市場資料來源。

FinMind 的每個 dataset 都同時收錄上市與上櫃的股票,格式相同,
所以這一層不必知道股票掛在哪個市場。
"""

from dataclasses import dataclass
from datetime import date

import httpx2
import pandas as pd

from app.market import DAILY_DTYPES, QuotaExceededError, SourceUnavailableError

URL = "https://api.finmindtrade.com/api/v4/data"

# FinMind 的融資餘額以張為單位,一張是 1000 股
_SHARES_PER_LOT = 1000


@dataclass(frozen=True)
class FinMindSource:
    """從 FinMind 取得價格與籌碼,合成標準每日資料表。

    token 是選配;沒有 token(或是空字串)時,用 FinMind 給未註冊使用者的額度。
    transport 只給測試換成假的 HTTP 回應,正式使用時保持 None。
    """

    token: str | None = None
    transport: httpx2.BaseTransport | None = None

    def daily(self, stock_id: str, start: date, end: date) -> pd.DataFrame:
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        with httpx2.Client(
            transport=self.transport, timeout=60, headers=headers
        ) as client:

            def fetch(dataset: str, columns: list[str]) -> pd.DataFrame:
                return _fetch(client, dataset, stock_id, columns, start, end)

            prices = fetch(
                "TaiwanStockPrice",
                ["open", "max", "min", "close", "Trading_Volume", "Trading_money"],
            )
            margin = fetch(
                "TaiwanStockMarginPurchaseShortSale", ["MarginPurchaseTodayBalance"]
            )
            holding = fetch("TaiwanStockShareholding", ["ForeignInvestmentShares"])
            institutional = fetch(
                "TaiwanStockInstitutionalInvestorsBuySell", ["name", "buy", "sell"]
            )

        days = prices.index
        margin_balance = margin["MarginPurchaseTodayBalance"].reindex(days)
        foreign_shares = holding["ForeignInvestmentShares"].reindex(days)
        # 三大法人的買賣只列出當天有法人交易的股票,沒列出的日子就是買賣超為 0。
        # 但價格公布後、三大法人公布前的那段時間,當天同樣沒有資料。依 FinMind 的文件,
        # 三大法人比融資與外資持股早公布,所以融資或外資持股有任一項的日子,
        # 三大法人就已經公布
        published = margin_balance.notna() | foreign_shares.notna()

        def net_buy(*names: str) -> pd.Series:
            rows = institutional[institutional["name"].isin(names)]
            reported = (rows["buy"] - rows["sell"]).groupby(level="date").sum()
            reported = reported.reindex(days)
            return reported.mask(reported.isna() & published, 0)

        table = pd.DataFrame(
            {
                "open": prices["open"],
                "high": prices["max"],
                "low": prices["min"],
                "close": prices["close"],
                "volume": prices["Trading_Volume"],
                "trading_value": prices["Trading_money"],
                "margin_balance": margin_balance * _SHARES_PER_LOT,
                "foreign_shares": foreign_shares,
                "trust_net_buy": net_buy("Investment_Trust"),
                # 自營商分成自行買賣與避險兩類,合起來才是自營商的買賣超。
                # Foreign_Dealer_Self 是外資自營商,屬於外資,不算在這裡
                "dealer_net_buy": net_buy("Dealer_self", "Dealer_Hedging"),
            },
            index=days,
        )
        return table.astype(DAILY_DTYPES).sort_index()


def _fetch(
    client: httpx2.Client,
    dataset: str,
    stock_id: str,
    columns: list[str],
    start: date,
    end: date,
) -> pd.DataFrame:
    """回傳一個 dataset 在 start 到 end 之間的資料,只留 columns,以交易日為 index。"""
    params = {
        "dataset": dataset,
        "data_id": stock_id,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
    }
    try:
        response = client.get(URL, params=params)
    except httpx2.HTTPError as error:
        raise SourceUnavailableError(f"連不上 FinMind,請稍後再試({error})") from error
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
    records = pd.DataFrame(data, columns=["date", *columns])
    return records.set_index(pd.DatetimeIndex(records["date"], name="date"))[columns]


def _reason(response: httpx2.Response) -> str:
    """回傳 FinMind 在錯誤回應裡附的說明。

    回應不是 FinMind 的格式時(例如中間的 proxy 回的錯誤頁),回傳 HTTP 的狀態說明。
    """
    try:
        return response.json()["msg"]
    except (ValueError, KeyError, TypeError):
        return response.reason_phrase
