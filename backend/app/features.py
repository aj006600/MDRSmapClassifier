"""把標準每日資料表轉成演算法吃的特徵表:價量加上論文使用的技術指標。

技術指標的算法照論文的實作,包括論文對 MACD 而不是收盤價算 RSI 的做法。
"""

import pandas as pd


def feature_table(daily: pd.DataFrame) -> pd.DataFrame:
    """回傳演算法的特徵表,index 與 daily 相同,是交易日。

    欄位依序是開盤、最高、最低、收盤、成交量,接著是 EMA9、EMA12、EMA26、MACD、
    Signal 與 RSI14。欄位順序會影響 Search 的結果。

    每一列只用到當天(含)以前的資料。開頭幾十個交易日的技術指標還算不出來,
    這些列會被丟掉。有欄位缺值的交易日也會被丟掉,例如資料還沒公布完的最新交易日。
    """
    close = daily["close"]
    ema12 = _ema(close, 12)
    ema26 = _ema(close, 26)
    macd = ema12 - ema26
    return daily.assign(
        EMA9=_ema(close, 9),
        EMA12=ema12,
        EMA26=ema26,
        MACD=macd,
        Signal=_ema(macd, 9),
        RSI14=_rsi(macd, 14),
    ).dropna()


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, min_periods=span).mean()


def _rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    return 100 - (100 / (1 + gain / loss))
