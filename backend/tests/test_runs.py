"""一次執行的對外行為:階段、Search 用到的資料、結果的重現與 Direction 的標記。"""

import multiprocessing

import pandas as pd

from app.runs import run


def _as_of(daily: pd.DataFrame):
    return daily.index[-1].date()


def test_run_reports_stages_in_order(daily, run_config):
    stages = []
    run(daily, 10, _as_of(daily), config=run_config, on_stage=stages.append)
    assert stages == ["search", "predict"]


def test_run_with_previous_result_skips_search(daily, run_config):
    earlier = daily.iloc[:-5]
    previous = run(earlier, 10, _as_of(earlier), config=run_config)

    stages = []
    run(daily, 10, _as_of(daily), previous, run_config, on_stage=stages.append)

    assert stages == ["predict"]


def test_search_ignores_data_after_start(daily, run_config):
    original = run(daily, 10, _as_of(daily), config=run_config)

    changed = daily.copy()
    after_start = changed.index > pd.Timestamp(original.start)
    prices = ["open", "high", "low", "close"]
    changed.loc[after_start, prices] *= 1.1
    changed.loc[after_start, changed.columns.drop(prices)] *= 2
    rerun = run(changed, 10, _as_of(changed), config=run_config)

    assert rerun.start == original.start
    assert rerun.params == original.params


def test_run_ignores_gap_in_data_it_does_not_use(daily, run_config):
    # 執行拿到這支股票存下的所有日子,特徵只從起點往前 Library 加上
    # 技術指標開頭算不出來的日子算起。往前 90 個交易日以外,資料存了但用不到
    used = run_config.library_days + run_config.lookback_days + 1
    changed = daily.copy()
    gap = changed.index[-(used + 90) : -(used + 87)]
    changed.loc[gap, "foreign_shares"] = pd.NA

    rerun = run(changed, 10, _as_of(changed), config=run_config)

    pd.testing.assert_frame_equal(
        rerun.days, run(daily, 10, _as_of(daily), config=run_config).days
    )


def test_run_in_another_process_predicts_the_same_days(daily, run_config):
    args = (daily, 10, _as_of(daily), None, run_config)
    # spawn 的子程序是新的直譯器,連 hash 的亂數種子都和測試的程序不同
    with multiprocessing.get_context("spawn").Pool(1) as pool:
        spawned = pool.apply(run, args)
    pd.testing.assert_frame_equal(spawned.days, run(*args).days)


def test_actual_direction_compares_future_mean_with_close(
    known_directions_source, run_config
):
    daily = known_directions_source.table
    result = run(daily, 10, _as_of(daily), config=run_config)

    # 手算的過程寫在 known_directions_source
    expected = [0] + [1] * 10 + [pd.NA] * 10
    pd.testing.assert_series_equal(
        result.days["actual"],
        pd.Series(expected, index=result.days.index, dtype="Int64", name="actual"),
    )
