"""後端 HTTP API 的對外行為。

測試環境與正式環境有三點不同:

- 資料來源換成假實作,回傳固定的資料,或模擬資料來源的各種狀況
- 演算法的設定調小
- 每個測試用自己的暫時資料庫

執行在另一個程序裡跑,所以發起之後要輪詢工作直到結束。
"""

import time

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.market import DAILY_DTYPES, QuotaExceededError, SourceUnavailableError
from app.runs import run
from tests.fakes import DyingSource, FailingSource, FixedSource, HangingSource


@pytest.fixture
def database(tmp_path):
    return tmp_path / "mdrsmap.db"


@pytest.fixture
def client(fake_source, database, run_config):
    with TestClient(create_app(fake_source, database, run_config)) as client:
        yield client


@pytest.fixture
def hanging_client(database, run_config):
    """資料來源一直不回應的 app:第一個工作停在執行中,之後的工作停在排隊中。"""
    with TestClient(create_app(HangingSource(), database, run_config)) as client:
        yield client


def _start(client: TestClient, stock_id: str, horizon: int = 10) -> int:
    """發起一次執行,回傳工作的 id。"""
    response = client.post("/api/jobs", json={"stock_id": stock_id, "horizon": horizon})
    return response.json()["id"]


def _wait_until(
    client: TestClient, job_id: int, statuses: tuple[str, ...], timeout: float = 180
) -> dict:
    """輪詢工作直到它的狀態是 statuses 之一,回傳最後一次查到的工作。"""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in statuses:
            return job
        time.sleep(0.2)
    raise AssertionError(f"工作在 {timeout} 秒內沒有進入 {statuses}:{job}")


def _wait_until_done(client: TestClient, job_id: int) -> dict:
    return _wait_until(client, job_id, ("succeeded", "failed"))


def _run_job(source, database, run_config, stock_id: str = "2330") -> dict:
    """以 source 當資料來源發起一次執行,回傳結束時的工作。"""
    with TestClient(create_app(source, database, run_config)) as client:
        return _wait_until_done(client, _start(client, stock_id))


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_job_runs_through_stages_to_backtest_and_forecast(client, daily, run_config):
    response = client.post("/api/jobs", json={"stock_id": "2330", "horizon": 10})
    assert response.status_code == 202
    job = response.json()
    assert job["status"] == "queued"

    ended = _wait_until_done(client, job["id"])
    assert (ended["status"], ended["stage"]) == ("succeeded", "predict")
    result = client.get("/api/results/2330/10").json()

    # 從起點到最新交易日共 lookback_days + 1 天,最後 Horizon 天是 Forecast,其餘計分
    assert result["backtest"]["scored_days"] == run_config.lookback_days + 1 - 10
    assert [day["date"] for day in result["forecast"]] == [
        day.date().isoformat() for day in daily.index[-10:]
    ]


def test_job_without_horizon_uses_10(database, run_config):
    # 這裡只看發起時的 Horizon;用一定失敗的資料來源,工作才會很快結束
    source = FailingSource(ConnectionError("不需要資料"))
    with TestClient(create_app(source, database, run_config)) as client:
        job = client.post("/api/jobs", json={"stock_id": "2330"}).json()
        _wait_until_done(client, job["id"])

    assert job["horizon"] == 10


def test_result_reports_backtest_and_forecast_of_the_run(
    client, fake_source, run_config
):
    job = client.post("/api/jobs", json={"stock_id": "2330", "horizon": 10}).json()
    _wait_until_done(client, job["id"])
    result = client.get("/api/results/2330/10").json()

    # 在測試的程序裡用同樣的資料與設定再執行一次
    days = run(fake_source, "2330", 10, run_config).days
    scored = days.dropna()
    assert result["backtest"]["accuracy"] == (
        (scored["direction"] == scored["actual"]).sum() / len(scored)
    )
    assert result["backtest"]["days"] == [
        {
            "date": day.date().isoformat(),
            "direction": int(row.direction),
            "actual": int(row.actual),
        }
        for day, row in scored.iterrows()
    ]
    assert [day["direction"] for day in result["forecast"]] == days.loc[
        days["actual"].isna(), "direction"
    ].tolist()


def test_result_reports_the_last_trading_day_of_the_data(client, daily):
    job = client.post("/api/jobs", json={"stock_id": "2330", "horizon": 10}).json()
    _wait_until_done(client, job["id"])
    result = client.get("/api/results/2330/10").json()

    assert result["data_date"] == daily.index[-1].date().isoformat()


def test_direction_one_ratio_is_share_of_backtest_days_with_direction_1(
    known_directions_source, database, run_config
):
    app = create_app(known_directions_source, database, run_config)
    with TestClient(app) as client:
        job = client.post("/api/jobs", json={"stock_id": "2330", "horizon": 10}).json()
        _wait_until_done(client, job["id"])
        backtest = client.get("/api/results/2330/10").json()["backtest"]

    # 手算的過程寫在 known_directions_source:計分的 11 天裡只有第 0 天是 0
    assert backtest["direction_one_ratio"] == 10 / 11


def test_results_list_summarizes_each_stock_and_horizon_by_stock_id(client):
    # 假資料來源不論股票代號都回傳同一份資料,這裡只看清單怎麼列
    runs = [("2454", 10), ("2330", 10)]
    started = [
        client.post("/api/jobs", json={"stock_id": s, "horizon": h}).json()
        for s, h in runs
    ]
    for job in started:
        _wait_until_done(client, job["id"])

    listed = client.get("/api/results").json()

    expected = []
    for stock_id, horizon in sorted(runs):
        result = client.get(f"/api/results/{stock_id}/{horizon}").json()
        backtest = result["backtest"]
        expected.append(
            {
                "stock_id": stock_id,
                "horizon": horizon,
                "data_date": result["data_date"],
                "latest_forecast": result["forecast"][-1],
                "backtest": {
                    "accuracy": backtest["accuracy"],
                    "scored_days": backtest["scored_days"],
                    "direction_one_ratio": backtest["direction_one_ratio"],
                },
            }
        )
    assert listed == expected


def test_results_list_leaves_out_stock_whose_runs_failed(database, run_config):
    source = FailingSource(SourceUnavailableError("連不上 FinMind"))
    with TestClient(create_app(source, database, run_config)) as client:
        job = client.post("/api/jobs", json={"stock_id": "2330"}).json()
        _wait_until_done(client, job["id"])

        assert client.get("/api/results").json() == []


def test_response_schemas_have_no_daily_market_columns(client):
    # 資料來源的條款不允許把原始市場資料轉給第三方,API 只回傳預測與統計
    schemas = client.get("/openapi.json").json()["components"]["schemas"]
    fields = {
        name for schema in schemas.values() for name in schema.get("properties", {})
    }
    assert fields.isdisjoint(DAILY_DTYPES)


def test_result_is_kept_after_restart(fake_source, database, run_config):
    with TestClient(create_app(fake_source, database, run_config)) as client:
        job = client.post("/api/jobs", json={"stock_id": "2330"}).json()
        _wait_until_done(client, job["id"])
        before = client.get("/api/results/2330/10").json()

    with TestClient(create_app(fake_source, database, run_config)) as client:
        assert client.get("/api/results/2330/10").json() == before


def test_result_of_stock_never_run_is_not_found(client):
    assert client.get("/api/results/2330/10").status_code == 404


def test_unknown_job_is_not_found(client):
    assert client.get("/api/jobs/1").status_code == 404


def test_job_rejects_horizon_outside_choices(client):
    response = client.post("/api/jobs", json={"stock_id": "2330", "horizon": 7})
    assert response.status_code == 422


def test_jobs_in_progress_are_listed_in_the_order_they_run(hanging_client):
    ids = [_start(hanging_client, stock_id) for stock_id in ("2330", "2317", "2454")]
    _wait_until(hanging_client, ids[0], ("running",))

    jobs = hanging_client.get("/api/jobs").json()

    assert [(job["id"], job["status"]) for job in jobs] == [
        (ids[0], "running"),
        (ids[1], "queued"),
        (ids[2], "queued"),
    ]


def test_jobs_run_one_at_a_time_in_the_order_started(client):
    ids = [_start(client, stock_id) for stock_id in ("2330", "2317", "2454")]

    # 輪詢排隊中與執行中的工作,直到全部結束,記下依序看到執行中的是哪個工作
    running_order = []
    deadline = time.monotonic() + 300
    while jobs := client.get("/api/jobs").json():
        running = [job["id"] for job in jobs if job["status"] == "running"]
        assert len(running) <= 1, jobs
        if running and running_order[-1:] != running:
            running_order += running
        assert time.monotonic() < deadline, jobs
        time.sleep(0.1)

    assert running_order == ids
    statuses = [client.get(f"/api/jobs/{job_id}").json()["status"] for job_id in ids]
    assert statuses == ["succeeded"] * 3


def test_starting_a_stock_and_horizon_in_progress_returns_the_same_job(
    hanging_client,
):
    running = _start(hanging_client, "2330")
    queued = _start(hanging_client, "2317")
    _wait_until(hanging_client, running, ("running",))

    assert _start(hanging_client, "2330") == running
    assert _start(hanging_client, "2317") == queued
    # 同一支股票換一個 Horizon 是另一個工作
    assert _start(hanging_client, "2330", horizon=5) not in (running, queued)


def test_restart_fails_the_running_job_and_runs_the_queued_one(
    fake_source, database, run_config
):
    with TestClient(create_app(HangingSource(), database, run_config)) as client:
        interrupted = _start(client, "2330")
        queued = _start(client, "2317")
        _wait_until(client, interrupted, ("running",))

    with TestClient(create_app(fake_source, database, run_config)) as client:
        ended = client.get(f"/api/jobs/{interrupted}").json()
        assert ended["status"] == "failed"
        assert "重新啟動" in ended["error"]
        assert _wait_until_done(client, queued)["status"] == "succeeded"


def test_job_whose_process_dies_fails_and_the_next_job_runs(database, run_config):
    with TestClient(create_app(DyingSource(), database, run_config)) as client:
        ids = [_start(client, "2330"), _start(client, "2317")]
        ended = [_wait_until_done(client, job_id) for job_id in ids]

    assert [job["status"] for job in ended] == ["failed", "failed"]
    assert "意外結束" in ended[0]["error"]


def test_unknown_stock_fails_saying_it_is_not_found(daily, database, run_config):
    # 資料來源沒有這支股票的任何一個交易日
    ended = _run_job(FixedSource(daily.iloc[:0]), database, run_config, "9999")

    assert ended["status"] == "failed"
    assert "查無股票代號 9999" in ended["error"]


def test_stock_with_short_history_fails_saying_how_much_is_needed(
    daily, database, run_config
):
    # 扣掉技術指標開頭算不出來的日子後,起點之前不到 library_days 個交易日
    ended = _run_job(FixedSource(daily.tail(150)), database, run_config)

    assert ended["status"] == "failed"
    assert "歷史資料不足" in ended["error"]
    assert f"{run_config.library_days} 個交易日" in ended["error"]


def test_exhausted_quota_fails_saying_to_wait_or_set_a_token(database, run_config):
    ended = _run_job(FailingSource(QuotaExceededError()), database, run_config)

    assert ended["status"] == "failed"
    assert "額度" in ended["error"]
    assert "稍後再試" in ended["error"]
    assert "FINMIND_TOKEN" in ended["error"]


def test_unreachable_source_fails_saying_the_source_has_a_problem(database, run_config):
    error = SourceUnavailableError("連不上 FinMind(Name or service not known)")
    ended = _run_job(FailingSource(error), database, run_config)

    assert ended["status"] == "failed"
    assert "資料來源連不上" in ended["error"]
    # 資料來源給的細節也要讓使用者看到
    assert "Name or service not known" in ended["error"]


def test_unexpected_error_fails_saying_it_is_a_program_error(database, run_config):
    ended = _run_job(FailingSource(ValueError("意料之外的值")), database, run_config)

    assert ended["status"] == "failed"
    assert "程式錯誤" in ended["error"]
    assert "ValueError: 意料之外的值" in ended["error"]


def test_incomplete_latest_day_is_left_out_and_result_names_the_day_used(
    daily, database, run_config
):
    # 最新交易日的收盤價還沒公布
    incomplete = daily.copy()
    incomplete.iloc[-1, incomplete.columns.get_loc("close")] = float("nan")

    with TestClient(
        create_app(FixedSource(incomplete), database, run_config)
    ) as client:
        _wait_until_done(client, _start(client, "2330"))
        result = client.get("/api/results/2330/10").json()

    last_complete_day = daily.index[-2].date().isoformat()
    assert result["data_date"] == last_complete_day
    assert result["forecast"][-1]["date"] == last_complete_day
