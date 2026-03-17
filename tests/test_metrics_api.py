from __future__ import annotations

from fastapi.testclient import TestClient


def test_metrics_endpoint_exposes_request_counters(client: TestClient) -> None:
    client.get("/health")

    response = client.get("/metrics")

    assert response.status_code == 200
    assert response.json()["request_count"] >= 1
