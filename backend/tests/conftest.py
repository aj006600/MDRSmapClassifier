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
def fake_source(daily) -> FixedSource:
    return FixedSource(daily)


@pytest.fixture(scope="session")
def run_config() -> RunConfig:
    """調小到一次執行幾秒內跑完的設定。結果不對應論文,只用來驗證行為。"""
    return RunConfig(
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
