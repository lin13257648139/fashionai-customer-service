from offline_api import create_app


def test_health_reports_offline_mode():
    client = create_app().test_client()
    response = client.get("/health")
    assert response.status_code == 200
    assert response.get_json()["mode"] == "offline"


def test_chat_works_without_external_api():
    client = create_app().test_client()
    response = client.post("/chat", json={"message": "你好", "session_id": "test"})
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["success"] is True
    assert payload["mode"] == "offline"


def test_chat_rejects_empty_message():
    client = create_app().test_client()
    response = client.post("/chat", json={"message": ""})
    assert response.status_code == 400
