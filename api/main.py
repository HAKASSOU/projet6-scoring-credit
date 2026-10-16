"""API de scoring crédit « Prêt à dépenser ».

Endpoints :
    GET  /health            état du service et version du modèle
    GET  /clients           liste d'identifiants clients disponibles
    POST /predict           score, décision et facteurs explicatifs d'un client

Les contributions des variables sont calculées nativement par LightGBM
(``pred_contrib=True``, équivalent aux valeurs SHAP d'un arbre) : pas de dépendance à shap.
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from pydantic import BaseModel, Field

logger = logging.getLogger("scoring_api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

DEFAULT_MODEL_DIR = Path(os.environ.get("MODEL_DIR", "models"))
DEFAULT_CLIENTS_PATH = Path(os.environ.get("CLIENTS_PATH", "data/clients_demo.parquet"))
TOP_FACTORS = 5


class PredictionRequest(BaseModel):
    """Corps de la requête /predict."""

    sk_id_curr: int = Field(..., gt=0, description="Identifiant du client (SK_ID_CURR)")


class Factor(BaseModel):
    """Contribution d'une variable au score (en log-odds)."""

    feature: str
    value: float | None
    impact: float


class PredictionResponse(BaseModel):
    """Réponse de /predict."""

    sk_id_curr: int
    probability_default: float
    threshold: float
    decision: str
    risk_factors: list[Factor]
    protective_factors: list[Factor]


class ScoringService:
    """Charge le modèle et les clients, et produit les scores."""

    def __init__(self, model_dir: Path, clients_path: Path) -> None:
        self.model = lgb.Booster(model_file=str(model_dir / "lgbm_model.txt"))
        meta = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
        self.threshold: float = float(meta["seuil_decision"])
        self.features: list[str] = list(meta["variables"])
        self.clients = pd.read_parquet(clients_path)
        missing = [c for c in self.features if c not in self.clients.columns]
        if missing:
            raise ValueError(f"Variables absentes des données clients : {missing[:5]}")
        self.clients = self.clients[self.features]
        logger.info(
            "Service prêt : %d clients, %d variables", len(self.clients), len(self.features)
        )

    def client_ids(self, limit: int) -> list[int]:
        """Premiers identifiants disponibles."""
        return [int(i) for i in self.clients.index[:limit]]

    def predict(self, client_id: int, top: int = TOP_FACTORS) -> PredictionResponse:
        """Score un client connu ; lève ``KeyError`` s'il n'existe pas."""
        if client_id not in self.clients.index:
            raise KeyError(client_id)
        row = self.clients.loc[[client_id]]
        contrib = self.model.predict(row, pred_contrib=True)[0][:-1]  # dernier = biais
        proba = float(self.model.predict(row)[0])
        impacts = pd.Series(contrib, index=self.features)
        values = row.iloc[0]

        def build(series: pd.Series) -> list[Factor]:
            return [
                Factor(
                    feature=name,
                    value=None if np.isnan(values[name]) else float(values[name]),
                    impact=round(float(impact), 4),
                )
                for name, impact in series.items()
            ]

        return PredictionResponse(
            sk_id_curr=client_id,
            probability_default=round(proba, 4),
            threshold=self.threshold,
            decision="REFUS" if proba >= self.threshold else "ACCORD",
            risk_factors=build(impacts[impacts > 0].nlargest(top)),
            protective_factors=build(impacts[impacts < 0].nsmallest(top)),
        )


def create_app(model_dir: Path | None = None, clients_path: Path | None = None) -> FastAPI:
    """Fabrique l'application (les chemins par défaut viennent des variables d'environnement)."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.service = ScoringService(
            model_dir or DEFAULT_MODEL_DIR, clients_path or DEFAULT_CLIENTS_PATH
        )
        yield

    app = FastAPI(title="Scoring crédit – Prêt à dépenser", version="1.0.0", lifespan=lifespan)

    @app.get("/health")
    def health(request: Request) -> dict:
        service: ScoringService = request.app.state.service
        return {
            "status": "ok",
            "threshold": service.threshold,
            "n_features": len(service.features),
            "n_clients": len(service.clients),
        }

    @app.get("/clients")
    def clients(request: Request, limit: int = Query(20, ge=1, le=1000)) -> dict:
        return {"sk_id_curr": request.app.state.service.client_ids(limit)}

    @app.post("/predict", response_model=PredictionResponse)
    def predict(body: PredictionRequest, request: Request) -> PredictionResponse:
        try:
            return request.app.state.service.predict(body.sk_id_curr)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Client {body.sk_id_curr} introuvable")

    return app


app = create_app()
