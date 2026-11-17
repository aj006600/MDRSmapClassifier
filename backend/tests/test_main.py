"""後端 HTTP API 的對外行為。

測試環境與正式環境有三點不同:

- 資料來源換成回傳固定資料的假實作
- 演算法的設定調小
- 每個測試用自己的暫時資料庫

執行在另一個程序裡跑,所以發起之後要輪詢工作直到結束。
"""

import time

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.runs import run
from tests.fakes import FailingSource


@pytest.fixture
def database(tmp_path):
    return tmp_path / "mdrsmap.db"


@pytest.fixture
def client(fake_source, database, run_config):
    with TestClient(create_app(fake_source, database, run_config)) as client:
        yield client


def _wait_until_done(client: TestClient, job_id: int, timeout: float = 180) -> dict:
    """輪詢工作直到成功或失敗,回傳最後一次查到的工作。"""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("succeeded", "failed"):
            return job
        time.sleep(0.2)
    raise AssertionError(f"工作在 {timeout} 秒內沒有結束:{job}")


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
    source = FailingSource("不需要資料")
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
    assert [day["direction"] for day in result["forecast"]] == days.loc[
        days["actual"].isna(), "direction"
    ].tolist()


def test_result_is_kept_after_restart(fake_source, database, run_config):
    with TestClient(create_app(fake_source, database, run_config)) as client:
        job = client.post("/api/jobs", json={"stock_id": "2330"}).json()
        _wait_until_done(client, job["id"])
        before = client.get("/api/results/2330/10").json()

    with TestClient(create_app(fake_source, database, run_config)) as client:
        assert client.get("/api/results/2330/10").json() == before


def test_failed_job_keeps_the_reason(database, run_config):
    source = FailingSource("FinMind 連不上")
    with TestClient(create_app(source, database, run_config)) as client:
        job = client.post("/api/jobs", json={"stock_id": "2330"}).json()
        ended = _wait_until_done(client, job["id"])

    assert ended["status"] == "failed"
    assert "FinMind 連不上" in ended["error"]


def test_result_of_stock_never_run_is_not_found(client):
    assert client.get("/api/results/2330/10").status_code == 404


def test_unknown_job_is_not_found(client):
    assert client.get("/api/jobs/1").status_code == 404


def test_job_rejects_horizon_outside_choices(client):
    response = client.post("/api/jobs", json={"stock_id": "2330", "horizon": 7})
    assert response.status_code == 422
