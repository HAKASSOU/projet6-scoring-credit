"""Exporte un petit échantillon de clients pour l'API et le dashboard.

Le fichier complet ``features_test.parquet`` (~150 Mo) est trop lourd pour GitHub :
on en extrait quelques milliers de lignes (quelques Mo).

Usage (depuis la racine du projet) :
    python scripts/export_demo_data.py
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

RAW_DIR = Path(
    os.environ.get(
        "P6_DATA_DIR",
        r"D:\Python_Datascience\Projet_OC6\Projet+Mise+en+prod+-+home-credit-default-risk",
    )
)
N_CLIENTS = int(os.environ.get("P6_N_DEMO", 2000))
SEED = 42


def main() -> None:
    """Échantillonne N_CLIENTS clients du jeu de test et les sauvegarde en parquet."""
    meta = json.loads(Path("models/metadata.json").read_text(encoding="utf-8"))
    clients = pd.read_parquet(RAW_DIR / "processed" / "features_test.parquet")
    sample = clients.sample(n=min(N_CLIENTS, len(clients)), random_state=SEED).sort_index()
    Path("data").mkdir(exist_ok=True)
    out = Path("data/clients_demo.parquet")
    sample[meta["variables"]].to_parquet(out)
    logger.info("%d clients -> %s (%.1f Mo)", len(sample), out, out.stat().st_size / 1e6)


if __name__ == "__main__":
    main()
