"""Tests du dashboard : libellés, formats et exécution de la page avec une API simulée."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from dashboard.labels import format_value, label

APP_PATH = Path(__file__).resolve().parent.parent / "dashboard" / "app.py"


def test_label_known_and_prefixed():
    assert label("EXT_SOURCE_2").startswith("Score externe n°2")
    assert label("INSTAL_DPD_MEAN") == "Échéances de remboursement – jours de retard (moyenne)"
    assert label("SOME_UNKNOWN_FEATURE") == "Some unknown feature"


def test_format_value():
    assert format_value("DAYS_BIRTH", -365.25 * 40) == "40,0 ans"
    assert format_value("AMT_CREDIT", 1234567.0) == "1 234 567"
    assert format_value("EXT_SOURCE_1", float("nan")) == "non renseigné"
    assert format_value("EXT_SOURCE_1", None) == "non renseigné"
    assert format_value("PAYMENT_RATE", 0.0512) == "0,051"


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


def test_app_runs_with_fake_api(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    n = 60
    clients = pd.DataFrame(
        {
            "DAYS_BIRTH": -rng.integers(8000, 24000, n).astype(float),
            "DAYS_EMPLOYED": -rng.integers(100, 9000, n).astype(float),
            "AMT_INCOME_TOTAL": rng.lognormal(12, 0.4, n),
            "AMT_CREDIT": rng.lognormal(13, 0.4, n),
            "AMT_ANNUITY": rng.lognormal(10, 0.4, n),
            "EXT_SOURCE_2": rng.random(n),
        },
        index=pd.Index(np.arange(1000, 1000 + n), name="SK_ID_CURR"),
    )
    path = tmp_path / "clients.parquet"
    clients.to_parquet(path)
    payload = {
        "sk_id_curr": 1000, "probability_default": 0.12, "threshold": 0.09,
        "decision": "REFUS",
        "risk_factors": [{"feature": "AMT_ANNUITY", "value": 25000.0, "impact": 0.3}],
        "protective_factors": [{"feature": "EXT_SOURCE_2", "value": None, "impact": -0.4}],
    }
    monkeypatch.setenv("CLIENTS_PATH", str(path))
    monkeypatch.setattr("requests.request", lambda *a, **k: FakeResponse(payload))
    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
    assert not app.exception
    assert any("REFUS" in e.value for e in app.error)
