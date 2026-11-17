"""一次執行的對外行為:階段、Search 用到的資料、結果的重現與 Direction 的標記。"""

import multiprocessing

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
    prices = ["open", "high", "low", "close"]
    changed.loc[after_start, prices] *= 1.1
    changed.loc[after_start, changed.columns.drop(prices)] *= 2
    rerun = run(FixedSource(changed), "2330", 10, run_config)

    assert rerun.start == original.start
    assert rerun.params == original.params


def test_run_ignores_gap_in_data_it_does_not_use(fake_source, daily, run_config):
    # 執行抓回來的日子比 Library 加上回溯的日子多出一大段,留給技術指標與長假。
    # 往前 90 個交易日以外,資料抓得到但用不到
    used = run_config.library_days + run_config.lookback_days + 1
    changed = daily.copy()
    gap = changed.index[-(used + 90) : -(used + 87)]
    changed.loc[gap, "foreign_shares"] = pd.NA

    rerun = run(FixedSource(changed), "2330", 10, run_config)

    pd.testing.assert_frame_equal(
        rerun.days, run(fake_source, "2330", 10, run_config).days
    )


def test_run_in_another_process_predicts_the_same_days(fake_source, run_config):
    # spawn 的子程序是新的直譯器,連 hash 的亂數種子都和測試的程序不同
    with multiprocessing.get_context("spawn").Pool(1) as pool:
        spawned = pool.apply(run, (fake_source, "2330", 10, run_config))
    pd.testing.assert_frame_equal(
        spawned.days, run(fake_source, "2330", 10, run_config).days
    )


def test_actual_direction_compares_future_mean_with_close(
    known_directions_source, run_config
):
    result = run(known_directions_source, "2330", 10, run_config)

    # 手算的過程寫在 known_directions_source
    expected = [0] + [1] * 10 + [pd.NA] * 10
    pd.testing.assert_series_equal(
        result.days["actual"],
        pd.Series(expected, index=result.days.index, dtype="Int64", name="actual"),
    )
