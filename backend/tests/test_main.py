from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_message_is_served_under_api_prefix():
    # 前端的 nginx 只把 /api/ 代理到後端,頁面靠這個端點證明前後端接通
    response = client.get("/api/message")
    assert response.status_code == 200
    assert response.json()["message"]
