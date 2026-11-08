from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from pipeline.config import FEATURE_COLS
from serving.schemas import RiskPredictionRequest


@pytest.fixture(scope="module")
def client(lakehouse):
    from serving.main import app

    with TestClient(app) as c:
        yield c


def test_request_schema_matches_feature_list():
    assert [f for f in RiskPredictionRequest.model_fields if f != "symbol"] == FEATURE_COLS


def test_health_and_symbols(client):
    h = client.get("/health").json()
    assert h == {"status": "ok", "model_loaded": True, "silver_available": True}
    assert "AAA.NS" in client.get("/api/v1/symbols").json()


def test_predict_roundtrip_and_validation(client):
    f = client.get("/api/v1/latest-features", params={"symbol": "AAA.NS"}).json()
    r = client.post("/api/v1/predict-risk", json={"symbol": "AAA.NS", **f["features"]})
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["risk_score"] <= 1
    assert body["risk_level"] in {"LOW", "ELEVATED", "CRITICAL"}
    assert body["latency_ms"] < 50
    bad = client.post("/api/v1/predict-risk", json={"symbol": "AAA.NS", **{**f["features"], "rsi_14": 400}})
    assert bad.status_code == 422
    missing = {k: v for k, v in f["features"].items() if k != "atr_ratio"}
    assert client.post("/api/v1/predict-risk", json={"symbol": "AAA.NS", **missing}).status_code == 422
    assert client.get("/api/v1/latest-features", params={"symbol": "NOPE"}).status_code == 404


def test_risk_tiers_follow_thresholds(client, lakehouse):
    from serving.main import _tier

    t = lakehouse["meta"]["thresholds"]
    assert _tier(t["elevated"] - 1e-6) == "LOW"
    assert _tier(t["elevated"]) in {"ELEVATED", "CRITICAL"}
    assert _tier(t["critical"]) == "CRITICAL"


def test_model_info_and_drift(client):
    info = client.get("/api/v1/model-info").json()
    assert info["n_features"] == len(FEATURE_COLS) and info["requests_served"] >= 1
    d = client.get("/api/v1/drift-status").json()
    assert d["source"] == "market" and set(d["feature_details"]) == set(FEATURE_COLS)
    assert client.get("/api/v1/drift-status", params={"source": "live"}).status_code == 409
    assert client.get("/api/v1/drift-status", params={"source": "bogus"}).status_code == 422
