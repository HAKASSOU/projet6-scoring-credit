"""Tests de l'API de scoring (modèle jouet entraîné à la volée)."""

from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import create_app

FEATURES = ["EXT_SOURCE_2", "AMT_ANNUITY", "DAYS_EMPLOYED"]


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """Client HTTP de test branché sur un petit modèle et 50 clients fictifs."""
    root = tmp_path_factory.mktemp("assets")
    rng = np.random.default_rng(0)
    data = pd.DataFrame(rng.normal(size=(500, 3)), columns=FEATURES).astype(np.float32)
    data.index = pd.Index(np.arange(1000, 1500), name="SK_ID_CURR")
    data.iloc[0, 2] = np.nan  # une valeur manquante
    label = (data["EXT_SOURCE_2"].fillna(0) + rng.normal(size=500) < -0.5).astype(int)
    booster = lgb.train(
        {"objective": "binary", "verbose": -1, "min_data_in_leaf": 5},
        lgb.Dataset(data, label=label),
        num_boost_round=20,
    )
    booster.save_model(str(root / "lgbm_model.txt"))
    meta = {"seuil_decision": 0.2, "variables": FEATURES}
    (root / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    data.head(50).to_parquet(root / "clients.parquet")
    with TestClient(create_app(root, root / "clients.parquet")) as test_client:
        yield test_client


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["n_clients"] == 50 and body["threshold"] == 0.2


def test_clients_list(client):
    ids = client.get("/clients", params={"limit": 3}).json()["sk_id_curr"]
    assert ids == [1000, 1001, 1002]


def test_predict_ok(client):
    response = client.post("/predict", json={"sk_id_curr": 1000})
    assert response.status_code == 200
    body = response.json()
    assert 0 <= body["probability_default"] <= 1
    assert body["decision"] == ("REFUS" if body["probability_default"] >= 0.2 else "ACCORD")
    assert all(f["impact"] > 0 for f in body["risk_factors"])
    assert all(f["impact"] < 0 for f in body["protective_factors"])


def test_predict_client_with_missing_value(client):
    """Le client 1000 a une valeur manquante : la réponse doit rester du JSON valide."""
    response = client.post("/predict", json={"sk_id_curr": 1000})
    assert response.status_code == 200
    json.loads(response.text)


def test_predict_unknown_client(client):
    assert client.post("/predict", json={"sk_id_curr": 42}).status_code == 404


def test_predict_invalid_payload(client):
    assert client.post("/predict", json={"sk_id_curr": -1}).status_code == 422
    assert client.post("/predict", json={}).status_code == 422
