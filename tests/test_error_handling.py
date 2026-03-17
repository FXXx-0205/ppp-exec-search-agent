from __future__ import annotations

from fastapi.testclient import TestClient


def test_validation_errors_follow_standard_shape(client: TestClient) -> None:
    response = client.post("/search/intake", json={"raw_input": ""})

    body = response.json()
    assert response.status_code == 422
    assert body["error"]["code"] == "request_validation_error"
    assert body["error"]["request_id"].startswith("req_")
