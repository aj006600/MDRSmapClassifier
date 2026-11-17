"""把標準每日資料表轉成演算法吃的特徵表:價量與籌碼,加上論文使用的技術指標。

技術指標的算法照論文的實作,包括論文對 MACD 而不是收盤價算 RSI 的做法。
"""

import pandas as pd

# 特徵用到的標準每日資料表欄位
_PRICE_COLUMNS = ["open", "high", "low", "close", "volume"]
_BALANCE_COLUMNS = ["margin_balance", "foreign_shares"]
_CHIP_COLUMNS = [*_BALANCE_COLUMNS, "trust_net_buy", "dealer_net_buy"]


class IncompleteDataError(ValueError):
    """每日資料表在兩個資料齊全的交易日之間有補不了的缺值。"""


def feature_table(daily: pd.DataFrame) -> pd.DataFrame:
    """回傳演算法的特徵表,index 是 daily 的交易日。

    欄位依序如下,欄位順序會影響 Search 的結果:

    - 開盤、最高、最低、收盤、成交量
    - margin_change、foreign_change:融資餘額與外資持股的日差
    - trust_net_buy、dealer_net_buy:投信與自營商的買賣超
    - EMA9、EMA12、EMA26、MACD、Signal 與 RSI14

    每一列只用到當天(含)以前的資料。開頭幾十個交易日的技術指標還算不出來,
    這些列會被丟掉。

    籌碼資料不齊全的交易日處理如下:

    - 剛上市的股票要過一段時間才有籌碼資料,特徵表從第一個資料齊全的交易日開始
    - 最新幾個交易日的籌碼資料可能還沒公布,特徵表只到最後一個資料齊全的交易日
    - 資料來源偶爾漏掉某一天的融資餘額或外資持股。只漏一天時沿用前一天的值,
      所以那天的日差是 0,隔天的日差包含兩天的變化
    - 其餘夾在中間的缺值丟出 IncompleteDataError。略過那幾天會讓 Horizon 數錯交易日,
      連續好幾天的餘額與當天的買賣超也沒有合理的值可以補
    """
    daily = _usable_span(daily[_PRICE_COLUMNS + _CHIP_COLUMNS])
    close = daily["close"]
    ema12 = _ema(close, 12)
    ema26 = _ema(close, 26)
    macd = ema12 - ema26
    chips = daily[_CHIP_COLUMNS].astype("float64")
    return pd.DataFrame(
        {
            "open": daily["open"],
            "high": daily["high"],
            "low": daily["low"],
            "close": close,
            "volume": daily["volume"],
            "margin_change": chips["margin_balance"].diff(),
            "foreign_change": chips["foreign_shares"].diff(),
            "trust_net_buy": chips["trust_net_buy"],
            "dealer_net_buy": chips["dealer_net_buy"],
            "EMA9": _ema(close, 9),
            "EMA12": ema12,
            "EMA26": ema26,
            "MACD": macd,
            "Signal": _ema(macd, 9),
            "RSI14": _rsi(macd, 14),
        }
    ).dropna()


def _usable_span(daily: pd.DataFrame) -> pd.DataFrame:
    """截掉頭尾不齊全的交易日,並補上餘額的單日缺漏。

    補完之後還有缺值時丟出 IncompleteDataError。
    """
    complete = daily.index[daily.notna().all(axis=1)]
    if complete.empty:
        raise IncompleteDataError("沒有任何一個交易日的資料是齊全的")
    daily = daily.loc[complete[0] : complete[-1]].copy()
    balances = daily[_BALANCE_COLUMNS]
    single = balances.isna() & balances.shift(1).notna() & balances.shift(-1).notna()
    daily[_BALANCE_COLUMNS] = balances.mask(single, balances.shift(1))

    missing = daily.isna()
    gaps = missing.loc[missing.any(axis=1), missing.any()]
    if not gaps.empty:
        raise IncompleteDataError(
            f"資料來源在 {gaps.index[0].date().isoformat()} 等 {len(gaps)} 個交易日"
            f"缺少 {'、'.join(gaps.columns)} 的資料。這些日子的前後都有資料齊全的"
            "交易日,所以不是還沒公布,也不是還沒上市"
        )
    return daily


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, min_periods=span).mean()


def _rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    return 100 - (100 / (1 + gain / loss))
