"""一次執行的對外行為:階段、Search 用到的資料、結果的重現與 Direction 的標記。"""

import multiprocessing
from dataclasses import replace

import pandas as pd

from app.runs import run
from tests.fakes import FixedSource


def test_run_reports_stages_in_order(fake_source, run_config):
    stages = []
    run(fake_source, "2330", 10, run_config, on_stage=stages.append)
    assert stages == ["fetch", "search", "predict"]


def test_search_ignores_data_after_start(fake_source, daily, run_config):
    original = run(fake_source, "2330", 10, run_config)

    changed = daily.copy()
    after_start = changed.index > pd.Timestamp(original.start)
    changed.loc[after_start, ["open", "high", "low", "close"]] *= 1.1
    changed.loc[after_start, "volume"] *= 2
    rerun = run(FixedSource(changed), "2330", 10, run_config)

    assert rerun.start == original.start
    assert rerun.params == original.params


def test_run_in_another_process_predicts_the_same_days(fake_source, run_config):
    # spawn 的子程序是新的直譯器,連 hash 的亂數種子都和測試的程序不同
    with multiprocessing.get_context("spawn").Pool(1) as pool:
        spawned = pool.apply(run, (fake_source, "2330", 10, run_config))
    pd.testing.assert_frame_equal(
        spawned.days, run(fake_source, "2330", 10, run_config).days
    )


def test_actual_direction_compares_future_mean_with_close(daily, run_config):
    # 這個例子固定從起點到最新交易日共 21 天,收盤價依序換成下面的值。
    # 以起點當天的收盤價 p 為基準,價位才不會離 Library 太遠
    config = replace(run_config, lookback_days=20)
    changed = daily.copy()
    close = changed.columns.get_loc("close")
    p = changed.iloc[-21, close]
    changed.iloc[-21:, close] = [p] * 10 + [p - 3] + [p + 3] * 10

    result = run(FixedSource(changed), "2330", 10, config)

    # 逐日手算未來 10 天的平均收盤價,與當天的收盤價比較:
    # - 第 0 天:未來 10 天含一天 p - 3,平均是 p - 0.3,低於當天的 p
    # - 第 1 天:未來 10 天含 p - 3 與 p + 3 各一天,平均剛好是 p,不低於當天
    # - 第 2 到 10 天:平均都高於當天
    # - 最後 10 天:未來的收盤價還沒出來
    expected = [0] + [1] * 10 + [pd.NA] * 10
    pd.testing.assert_series_equal(
        result.days["actual"],
        pd.Series(expected, index=result.days.index, dtype="Int64", name="actual"),
    )
