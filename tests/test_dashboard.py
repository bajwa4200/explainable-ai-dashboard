from xai.dashboard import create_app


def test_dashboard_routes():
    app = create_app()
    client = app.test_client()
    r = client.get("/")
    assert r.status_code == 200
    assert b"accuracy" in r.data.lower() or b"Accuracy" in r.data
    metrics = client.get("/api/metrics")
    assert metrics.status_code == 200
    data = metrics.get_json()
    assert "accuracy" in data
    assert len(data["importances"]) >= 1
