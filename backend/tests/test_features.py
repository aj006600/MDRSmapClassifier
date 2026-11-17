"""特徵表的對外行為。"""

import pandas as pd

from app.features import feature_table


def test_feature_table_uses_no_data_after_each_day(daily):
    # 任何一個特徵用到之後的資料,只給前段算出來的值就會和給全部算出來的不同
    last_day = daily.index[199]
    pd.testing.assert_frame_equal(
        feature_table(daily.loc[:last_day]),
        feature_table(daily).loc[:last_day],
        check_exact=True,
    )
