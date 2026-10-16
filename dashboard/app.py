"""Dashboard de scoring crédit pour les chargés de relation client.

Lancement (depuis la racine du projet) :
    streamlit run dashboard/app.py

Variables d'environnement :
    API_URL       adresse de l'API de scoring (défaut : http://127.0.0.1:8000)
    CLIENTS_PATH  fichier parquet des clients (défaut : data/clients_demo.parquet)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
from charts import distribution, factors_bar, gauge  # noqa: E402
from labels import format_value, label  # noqa: E402

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000").rstrip("/")
CLIENTS_PATH = Path(os.environ.get("CLIENTS_PATH", "data/clients_demo.parquet"))
PROFILE_VARIABLES = [
    "AGE", "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE",
    "DAYS_EMPLOYED", "CNT_CHILDREN", "CNT_FAM_MEMBERS", "EXT_SOURCE_1", "EXT_SOURCE_2",
    "EXT_SOURCE_3", "PAYMENT_RATE",
]
MARGIN = 0.02  # zone « dossier limite » autour du seuil (± 2 points)

st.set_page_config(page_title="Scoring crédit – Prêt à dépenser", layout="wide")


@st.cache_data
def load_clients(path: Path) -> pd.DataFrame:
    """Charge les clients et ajoute l'âge en années."""
    clients = pd.read_parquet(path)
    clients["AGE"] = (-clients["DAYS_BIRTH"] / 365.25).round(0)
    return clients


def call_api(method: str, endpoint: str, **kwargs) -> dict:
    """Appelle l'API ; affiche une erreur claire et arrête l'application en cas d'échec."""
    try:
        # timeout large : l'hébergement gratuit met l'API en veille (réveil ~ 1 minute)
        with st.spinner("Calcul du score en cours…"):
            response = requests.request(method, f"{API_URL}{endpoint}", timeout=90, **kwargs)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as error:
        st.error(
            f"Impossible de joindre l'API de scoring ({API_URL}). "
            "Vérifiez qu'elle est démarrée : `uvicorn api.main:app`."
        )
        st.caption(f"Détail technique : {error}")
        st.stop()


def similar_clients(clients: pd.DataFrame, row: pd.Series) -> pd.DataFrame:
    """Clients au profil proche : âge ± 5 ans, revenu et crédit ± 30 %."""
    mask = clients["AGE"].between(row["AGE"] - 5, row["AGE"] + 5)
    for column in ("AMT_INCOME_TOTAL", "AMT_CREDIT"):
        mask &= clients[column].between(row[column] * 0.7, row[column] * 1.3)
    return clients[mask]


def factors_frame(factors: list[dict]) -> pd.DataFrame:
    """Tableau des facteurs avec libellés et valeurs lisibles."""
    return pd.DataFrame(
        {
            "variable": [f["feature"] for f in factors],
            "libelle": [label(f["feature"]) for f in factors],
            "valeur": [format_value(f["feature"], f["value"]) for f in factors],
            "impact": [f["impact"] for f in factors],
        }
    )


def sidebar_filters(clients: pd.DataFrame) -> int:
    """Filtres de recherche d'un client ; retourne l'identifiant choisi."""
    st.sidebar.header("Trouver un client")
    age_min, age_max = int(clients["AGE"].min()), int(clients["AGE"].max())
    ages = st.sidebar.slider("Âge (ans)", age_min, age_max, (age_min, age_max))
    income_cap = float(clients["AMT_INCOME_TOTAL"].quantile(0.99))
    income = st.sidebar.slider(
        "Revenu annuel", 0, int(income_cap), (0, int(income_cap)), step=1000, format="%d"
    )
    credit_cap = float(clients["AMT_CREDIT"].quantile(0.99))
    credit = st.sidebar.slider(
        "Montant du crédit", 0, int(credit_cap), (0, int(credit_cap)), step=5000, format="%d"
    )
    mask = (
        clients["AGE"].between(*ages)
        & clients["AMT_INCOME_TOTAL"].between(*income)
        & clients["AMT_CREDIT"].between(*credit)
    )
    pool = clients[mask]
    st.sidebar.caption(f"{len(pool)} client(s) correspondent aux filtres.")
    if pool.empty:
        st.warning("Aucun client ne correspond à ces filtres : élargissez-les.")
        st.stop()
    return int(st.sidebar.selectbox("Numéro de client", pool.index.tolist()))


def show_decision(prediction: dict) -> None:
    """Jauge, décision et explication en langage simple."""
    probability, threshold = prediction["probability_default"], prediction["threshold"]
    left, right = st.columns([1, 1])
    with left:
        st.plotly_chart(gauge(probability, threshold))
    with right:
        if prediction["decision"] == "REFUS":
            st.error("### Décision proposée : REFUS ✖")
        else:
            st.success("### Décision proposée : ACCORD ✔")
        st.markdown(
            f"Ce client a une probabilité estimée de **{probability:.1%}** de ne pas "
            f"rembourser son crédit. Au-delà de **{threshold:.0%}**, la demande est orientée "
            f"vers un refus.\n\nAutrement dit, sur 100 clients au profil comparable, "
            f"environ **{probability * 100:.0f}** ne rembourseraient pas."
        )
        if abs(probability - threshold) <= MARGIN:
            st.warning("⚠ Dossier limite : le score est proche du seuil, un examen manuel est "
                       "recommandé.")
        st.caption("Aide à la décision : ce score ne remplace pas l'analyse du dossier.")


def show_explanation(prediction: dict) -> None:
    """Facteurs qui augmentent / réduisent le risque."""
    risk = factors_frame(prediction["risk_factors"])
    safe = factors_frame(prediction["protective_factors"])
    st.subheader("Pourquoi ce score ?")
    st.caption("Les barres montrent le poids de chaque information dans le score de ce client.")
    st.plotly_chart(factors_bar(pd.concat([risk, safe])))
    col_risk, col_safe = st.columns(2)
    with col_risk:
        st.markdown("**▲ Éléments qui augmentent le risque**")
        for _, item in risk.iterrows():
            st.markdown(f"- {item['libelle']} : **{item['valeur']}**")
    with col_safe:
        st.markdown("**▼ Éléments qui réduisent le risque**")
        for _, item in safe.iterrows():
            st.markdown(f"- {item['libelle']} : **{item['valeur']}**")


def show_profile(row: pd.Series) -> None:
    """Informations descriptives du client."""
    st.subheader("Profil du client")
    table = pd.DataFrame(
        {
            "Information": [label(v) for v in PROFILE_VARIABLES if v in row.index],
            "Valeur": [format_value(v, row[v]) for v in PROFILE_VARIABLES if v in row.index],
        }
    )
    st.table(table.set_index("Information"))


def show_comparison(clients: pd.DataFrame, row: pd.Series) -> None:
    """Position du client par rapport à tous les clients et aux clients similaires."""
    st.subheader("Comparaison avec les autres clients")
    similar = similar_clients(clients, row)
    st.caption(
        f"Clients similaires : âge ± 5 ans, revenu et crédit ± 30 % "
        f"({len(similar)} clients trouvés)."
    )
    options = [v for v in PROFILE_VARIABLES if v in clients.columns]
    variable = st.selectbox("Information à comparer", options, format_func=label)
    value = row[variable]
    value = None if pd.isna(value) else float(value)
    st.plotly_chart(distribution(clients[variable], similar[variable], value, label(variable)))

    rank = (
        float((clients[variable].dropna() <= value).mean() * 100) if value is not None else np.nan
    )
    summary = pd.DataFrame(
        {
            "Ce client": [format_value(variable, value)],
            "Médiane – tous": [format_value(variable, clients[variable].median())],
            "Médiane – similaires": [format_value(variable, similar[variable].median())],
            "Position parmi tous": [f"{rank:.0f}ᵉ percentile" if not np.isnan(rank) else "—"],
        }
    )
    st.table(summary.set_index("Ce client"))


def main() -> None:
    """Assemble la page."""
    st.title("Scoring crédit – Prêt à dépenser")
    if not CLIENTS_PATH.exists():
        st.error(f"Fichier clients introuvable : {CLIENTS_PATH}. "
                 "Lancez `python scripts/export_demo_data.py`.")
        st.stop()
    clients = load_clients(CLIENTS_PATH)
    client_id = sidebar_filters(clients)
    prediction = call_api("POST", "/predict", json={"sk_id_curr": client_id})
    row = clients.loc[client_id]

    st.markdown(f"#### Client n° {client_id}")
    show_decision(prediction)
    tab_why, tab_profile, tab_compare = st.tabs(
        ["Pourquoi ce score ?", "Profil du client", "Comparaison"]
    )
    with tab_why:
        show_explanation(prediction)
    with tab_profile:
        show_profile(row)
    with tab_compare:
        show_comparison(clients, row)


main()
