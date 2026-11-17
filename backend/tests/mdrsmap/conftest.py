"""準備演算法套件測試的輸入:論文的 2330 匯出資料,經過論文的特徵處理。

論文的資料讀取與特徵處理只用來產生回歸測試的輸入,所以留在測試裡,不進套件。
"""

from pathlib import Path

import pandas as pd
import pytest

DATA = Path(__file__).parent / "data"

# 匯出試算表的欄名對到論文程式用的欄名;欄位順序會影響 Search 的結果,不能調換
_PAPER_COLUMNS = {
    "交易日期": "Date",
    "開盤": "open",
    "最高": "high",
    "最低": "low",
    "收盤": "close",
    "成交量": "volume",
    "融資": "financing",
    "外資餘額": "fi",
    "投信餘額": "ii",
    "自營商買賣超": "di",
    "主力買賣超": "rp",
    "資金流向": "capital",
}


def load_paper_features() -> pd.DataFrame:
    """回傳論文 notebook 交給演算法的特徵表,以交易日為 index。"""
    raw = pd.read_excel(DATA / "2330.xlsx")
    table = raw[list(_PAPER_COLUMNS)].rename(columns=_PAPER_COLUMNS)
    table["Date"] = pd.to_datetime(table["Date"])
    table = table[table["Date"] >= "2020-11-17"]

    close = table["close"]
    table["EMA9"] = _ema(close, 9)
    table["EMA12"] = _ema(close, 12)
    table["EMA26"] = _ema(close, 26)
    table["MACD"] = table["EMA12"] - table["EMA26"]
    table["Signal"] = _ema(table["MACD"], 9)
    # 論文的實作對 MACD 而不是收盤價算 RSI,照抄才能重現論文
    table["RSI14"] = _rsi(table["MACD"], 14)

    balances = ["financing", "fi", "ii"]
    table[balances] = table[balances].diff()
    return table.dropna().set_index("Date")


@pytest.fixture(scope="session")
def paper_features() -> pd.DataFrame:
    return load_paper_features()


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, min_periods=span).mean()


def _rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    return 100 - (100 / (1 + gain / loss))
