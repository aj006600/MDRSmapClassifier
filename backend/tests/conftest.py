"""執行流程與 API 測試共用的準備:固定的每日資料與調小的設定。"""

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from app.market import DAILY_DTYPES
from app.runs import RunConfig
from mdrsmap import Config
from tests.fakes import FixedSource

PAPER_EXPORT = Path(__file__).parent / "mdrsmap" / "data" / "2330.xlsx"

# 匯出試算表的價量欄名對到標準每日資料表的欄名
_DAILY_COLUMNS = {
    "開盤": "open",
    "最高": "high",
    "最低": "low",
    "收盤": "close",
    "成交量": "volume",
}


@pytest.fixture(scope="session")
def daily() -> pd.DataFrame:
    """固定內容的標準每日資料表,最後一個交易日是今天。

    內容是論文 2330 匯出資料最後 300 個交易日的價量。執行以今天為準往前抓資料,
    所以日期換成到今天為止的平日。
    """
    raw = pd.read_excel(PAPER_EXPORT).tail(300)
    table = raw[list(_DAILY_COLUMNS)].rename(columns=_DAILY_COLUMNS)
    table = table.astype(DAILY_DTYPES).reset_index(drop=True)
    table.index = pd.bdate_range(end=date.today(), periods=len(table), name="date")
    return table


@pytest.fixture
def known_directions_source(daily) -> FixedSource:
    """最後 21 個交易日的收盤價換過、實際 Direction 可以手算的資料來源。

    搭配 run_config 執行時,起點到最新交易日剛好是這 21 天。以倒數第 21 天的
    收盤價 p 為基準,價位才不會離 Library 太遠。收盤價依序是 10 天 p、
    1 天 p - 3、10 天 p + 3。逐日手算 Horizon 10 的實際 Direction:

    - 第 0 天:未來 10 天含一天 p - 3,平均是 p - 0.3,低於當天的 p,所以是 0
    - 第 1 天:未來 10 天含 p - 3 與 p + 3 各一天,平均剛好是 p,不低於當天,所以是 1
    - 第 2 到 10 天:未來 10 天的平均都高於當天,所以是 1
    - 最後 10 天:未來的收盤價還沒出來,實際 Direction 還沒揭曉
    """
    changed = daily.copy()
    close = changed.columns.get_loc("close")
    p = changed.iloc[-21, close]
    changed.iloc[-21:, close] = [p] * 10 + [p - 3] + [p + 3] * 10
    return FixedSource(changed)


@pytest.fixture
def fake_source(daily) -> FixedSource:
    return FixedSource(daily)


@pytest.fixture(scope="session")
def run_config() -> RunConfig:
    """調小到一次執行幾秒內跑完的設定。結果不對應論文,只用來驗證行為。"""
    return RunConfig(
        # known_directions_source 手算的 21 天就是起點到最新交易日,改這個值要一起改
        lookback_days=20,
        library_days=100,
        algorithm=Config(
            top_views=5,
            max_embedding_dimension=2,
            max_lag=3,
            thetas=(1, 4),
            validation_days=20,
            random_search_trials=20,
        ),
    )
