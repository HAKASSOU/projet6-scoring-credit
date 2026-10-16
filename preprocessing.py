"""Pipeline de préparation des données Home Credit Default Risk.

Adapté du kernel « LightGBM with Simple Features » (J. Aguiar), réécrit pour :

* tenir dans 8 Go de RAM (float32 / uint8, lecture de bureau_balance par blocs) ;
* être reprenable : chaque table est sauvegardée en parquet et n'est recalculée
  que si on le demande (``force=True``) ;
* produire des types propres (aucune colonne ``object``) acceptés par LightGBM.

Utilisation::

    from preprocessing import run_pipeline, assemble_features
    run_pipeline(raw_dir, processed_dir)
    assemble_features(processed_dir)
"""

from __future__ import annotations

import gc
import logging
import re
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

KEY = "SK_ID_CURR"
DAYS_NAN_CODE = 365243  # valeur « sentinelle » utilisée pour les dates manquantes
BB_STATUSES = ["0", "1", "2", "3", "4", "5", "C", "X"]
BB_CHUNK_ROWS = 3_000_000

BINARY_MAPS = {
    "CODE_GENDER": {"M": 0, "F": 1},
    "FLAG_OWN_CAR": {"N": 0, "Y": 1},
    "FLAG_OWN_REALTY": {"N": 0, "Y": 1},
}


# --------------------------------------------------------------------------- #
# Utilitaires
# --------------------------------------------------------------------------- #
def clean_name(name: object) -> str:
    """Remplace tout caractère non alphanumérique par « _ » (noms acceptés par LightGBM)."""
    return re.sub(r"[^A-Za-z0-9_]", "_", str(name))


def downcast(df: pd.DataFrame) -> pd.DataFrame:
    """Convertit float64 -> float32 et int64 -> int32 pour réduire l'empreinte mémoire."""
    float_cols = df.select_dtypes("float64").columns
    if len(float_cols):
        df[float_cols] = df[float_cols].astype(np.float32)
    int_cols = df.select_dtypes("int64").columns
    if len(int_cols):
        df[int_cols] = df[int_cols].astype(np.int32)
    return df


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    """Lit un CSV puis réduit les types numériques."""
    logger.info("Lecture de %s", path.name)
    return downcast(pd.read_csv(path, **kwargs))


def one_hot_encode(
    df: pd.DataFrame, nan_as_category: bool = True
) -> tuple[pd.DataFrame, list[str]]:
    """Encode en one-hot (uint8) toutes les colonnes non numériques.

    Retourne le DataFrame encodé et la liste des nouvelles colonnes (noms nettoyés).
    Les indicatrices sont en ``uint8`` : ``min``/``max`` restent numériques après
    agrégation (avec des booléens, on obtenait des colonnes ``object``).
    """
    cat_cols = [
        c
        for c in df.columns
        if not pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])
    ]
    dummies = pd.get_dummies(df[cat_cols], dummy_na=nan_as_category, dtype=np.uint8)
    dummies.columns = [clean_name(c) for c in dummies.columns]
    out = pd.concat([df.drop(columns=cat_cols), dummies], axis=1)
    return out, list(dummies.columns)


def finalize(df: pd.DataFrame) -> pd.DataFrame:
    """Remplace ±inf par NaN et passe tout en float32."""
    return df.replace([np.inf, -np.inf], np.nan).astype(np.float32)


def aggregate(df: pd.DataFrame, spec: dict[str, list[str]], prefix: str) -> pd.DataFrame:
    """Agrège ``df`` par SK_ID_CURR et nomme les colonnes ``PREFIXE_colonne_STAT``."""
    agg = df.groupby(KEY).agg(spec)
    agg.columns = [f"{prefix}{col}_{stat.upper()}" for col, stat in agg.columns]
    return finalize(agg)


def _mean_spec(columns: list[str]) -> dict[str, list[str]]:
    """Spécification d'agrégation « moyenne » pour une liste de colonnes."""
    return {c: ["mean"] for c in columns}


# --------------------------------------------------------------------------- #
# Une fonction par table
# --------------------------------------------------------------------------- #
def process_application(raw_dir: Path) -> pd.DataFrame:
    """application_train + application_test : nettoyage et variables métier."""
    train = read_csv(raw_dir / "application_train.csv")
    test = read_csv(raw_dir / "application_test.csv")
    logger.info("application : train %s, test %s", train.shape, test.shape)
    df = pd.concat([train, test], ignore_index=True)  # train d'abord, puis test
    del train, test
    df["TARGET"] = df["TARGET"].astype(np.float32)

    df = df[df["CODE_GENDER"] != "XNA"].reset_index(drop=True)  # 4 lignes aberrantes
    for col, mapping in BINARY_MAPS.items():
        df[col] = df[col].map(mapping).astype(np.float32)

    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].where(df["DAYS_EMPLOYED"] != DAYS_NAN_CODE)
    df["DAYS_EMPLOYED_PERC"] = df["DAYS_EMPLOYED"] / df["DAYS_BIRTH"]
    df["INCOME_CREDIT_PERC"] = df["AMT_INCOME_TOTAL"] / df["AMT_CREDIT"]
    df["INCOME_PER_PERSON"] = df["AMT_INCOME_TOTAL"] / df["CNT_FAM_MEMBERS"]
    df["ANNUITY_INCOME_PERC"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
    df["PAYMENT_RATE"] = df["AMT_ANNUITY"] / df["AMT_CREDIT"]

    df, _ = one_hot_encode(df, nan_as_category=False)
    df = df.replace([np.inf, -np.inf], np.nan)
    return df.set_index(KEY)


def _bureau_balance_agg(path: Path, chunk_rows: int = BB_CHUNK_ROWS) -> pd.DataFrame:
    """Agrège bureau_balance (27 M lignes) par SK_ID_BUREAU, lecture par blocs."""
    logger.info("Lecture par blocs de %s", path.name)
    dtypes = {
        "SK_ID_BUREAU": np.int32,
        "MONTHS_BALANCE": np.int16,
        "STATUS": pd.CategoricalDtype(BB_STATUSES),
    }
    parts = []
    for chunk in pd.read_csv(path, dtype=dtypes, chunksize=chunk_rows):
        base = chunk.groupby("SK_ID_BUREAU")["MONTHS_BALANCE"].agg(["min", "max", "size"])
        counts = (
            chunk.groupby(["SK_ID_BUREAU", "STATUS"], observed=True)
            .size()
            .unstack(fill_value=0)
            .reindex(columns=BB_STATUSES, fill_value=0)
        )
        counts.columns = counts.columns.astype(str)
        parts.append(base.join(counts))
    merged = pd.concat(parts)
    rules = {"min": "min", "max": "max", "size": "sum", **{s: "sum" for s in BB_STATUSES}}
    total = merged.groupby(level=0).agg(rules)

    out = pd.DataFrame(index=total.index)
    out["MONTHS_BALANCE_MIN"] = total["min"]
    out["MONTHS_BALANCE_MAX"] = total["max"]
    out["MONTHS_BALANCE_SIZE"] = total["size"]
    for status in BB_STATUSES:
        out[f"STATUS_{status}_MEAN"] = total[status] / total["size"]
    return finalize(out)


def process_bureau(raw_dir: Path) -> pd.DataFrame:
    """bureau + bureau_balance : agrégats globaux, crédits actifs et clôturés."""
    bb_agg = _bureau_balance_agg(raw_dir / "bureau_balance.csv")
    bureau = read_csv(raw_dir / "bureau.csv")
    bureau, bureau_cat = one_hot_encode(bureau, nan_as_category=True)
    bureau = bureau.join(bb_agg, on="SK_ID_BUREAU")
    del bb_agg

    num_spec = {
        "DAYS_CREDIT": ["min", "max", "mean", "var"],
        "DAYS_CREDIT_ENDDATE": ["min", "max", "mean"],
        "DAYS_CREDIT_UPDATE": ["mean"],
        "CREDIT_DAY_OVERDUE": ["max", "mean"],
        "AMT_CREDIT_MAX_OVERDUE": ["mean"],
        "AMT_CREDIT_SUM": ["max", "mean", "sum"],
        "AMT_CREDIT_SUM_DEBT": ["max", "mean", "sum"],
        "AMT_CREDIT_SUM_OVERDUE": ["mean"],
        "AMT_CREDIT_SUM_LIMIT": ["mean", "sum"],
        "AMT_ANNUITY": ["max", "mean"],
        "CNT_CREDIT_PROLONG": ["sum"],
        "MONTHS_BALANCE_MIN": ["min"],
        "MONTHS_BALANCE_MAX": ["max"],
        "MONTHS_BALANCE_SIZE": ["mean", "sum"],
    }
    status_cols = [f"STATUS_{s}_MEAN" for s in BB_STATUSES]
    cat_spec = _mean_spec(bureau_cat + status_cols)

    result = aggregate(bureau, {**num_spec, **cat_spec}, "BURO_")
    for flag, prefix in (("CREDIT_ACTIVE_Active", "ACTIVE_"), ("CREDIT_ACTIVE_Closed", "CLOSED_")):
        subset = bureau[bureau[flag] == 1]
        result = result.join(aggregate(subset, num_spec, prefix), how="left")
    return result


def process_previous(raw_dir: Path) -> pd.DataFrame:
    """previous_application : agrégats globaux, demandes acceptées et refusées."""
    prev = read_csv(raw_dir / "previous_application.csv")
    prev, prev_cat = one_hot_encode(prev, nan_as_category=True)
    for col in (
        "DAYS_FIRST_DRAWING",
        "DAYS_FIRST_DUE",
        "DAYS_LAST_DUE_1ST_VERSION",
        "DAYS_LAST_DUE",
        "DAYS_TERMINATION",
    ):
        prev[col] = prev[col].where(prev[col] != DAYS_NAN_CODE)
    prev["APP_CREDIT_PERC"] = prev["AMT_APPLICATION"] / prev["AMT_CREDIT"]
    prev = prev.replace([np.inf, -np.inf], np.nan)

    num_spec = {
        "AMT_ANNUITY": ["min", "max", "mean"],
        "AMT_APPLICATION": ["min", "max", "mean"],
        "AMT_CREDIT": ["min", "max", "mean"],
        "APP_CREDIT_PERC": ["min", "max", "mean", "var"],
        "AMT_DOWN_PAYMENT": ["min", "max", "mean"],
        "AMT_GOODS_PRICE": ["min", "max", "mean"],
        "HOUR_APPR_PROCESS_START": ["min", "max", "mean"],
        "RATE_DOWN_PAYMENT": ["min", "max", "mean"],
        "DAYS_DECISION": ["min", "max", "mean"],
        "CNT_PAYMENT": ["mean", "sum"],
    }
    result = aggregate(prev, {**num_spec, **_mean_spec(prev_cat)}, "PREV_")
    for flag, prefix in (
        ("NAME_CONTRACT_STATUS_Approved", "APPROVED_"),
        ("NAME_CONTRACT_STATUS_Refused", "REFUSED_"),
    ):
        subset = prev[prev[flag] == 1]
        result = result.join(aggregate(subset, num_spec, prefix), how="left")
    return result


def process_pos(raw_dir: Path) -> pd.DataFrame:
    """POS_CASH_balance : historique des crédits point de vente / cash."""
    pos = read_csv(
        raw_dir / "POS_CASH_balance.csv",
        usecols=["SK_ID_CURR", "MONTHS_BALANCE", "NAME_CONTRACT_STATUS", "SK_DPD", "SK_DPD_DEF"],
        dtype={"NAME_CONTRACT_STATUS": "category"},
    )
    pos, pos_cat = one_hot_encode(pos, nan_as_category=True)
    spec = {
        "MONTHS_BALANCE": ["max", "mean", "size"],
        "SK_DPD": ["max", "mean"],
        "SK_DPD_DEF": ["max", "mean"],
        **_mean_spec(pos_cat),
    }
    result = aggregate(pos, spec, "POS_")
    result["POS_COUNT"] = pos.groupby(KEY).size().astype(np.float32)
    return result


def process_instalments(raw_dir: Path) -> pd.DataFrame:
    """installments_payments : retards / avances de paiement, écarts de montants."""
    float_cols = ["NUM_INSTALMENT_VERSION", "DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT"]
    float_cols += ["AMT_INSTALMENT", "AMT_PAYMENT"]
    ins = read_csv(
        raw_dir / "installments_payments.csv",
        usecols=[KEY, *float_cols],
        dtype={**{c: np.float32 for c in float_cols}, KEY: np.int32},
    )
    ins["PAYMENT_PERC"] = (ins["AMT_PAYMENT"] / ins["AMT_INSTALMENT"]).replace(
        [np.inf, -np.inf], np.nan
    )
    ins["PAYMENT_DIFF"] = ins["AMT_INSTALMENT"] - ins["AMT_PAYMENT"]
    delay = ins["DAYS_ENTRY_PAYMENT"] - ins["DAYS_INSTALMENT"]
    ins["DPD"] = delay.clip(lower=0).fillna(0)  # jours de retard
    ins["DBD"] = (-delay).clip(lower=0).fillna(0)  # jours d'avance
    del delay

    spec = {
        "NUM_INSTALMENT_VERSION": ["nunique"],
        "DPD": ["max", "mean", "sum"],
        "DBD": ["max", "mean", "sum"],
        "PAYMENT_PERC": ["max", "mean", "sum", "var"],
        "PAYMENT_DIFF": ["max", "mean", "sum", "var"],
        "AMT_INSTALMENT": ["max", "mean", "sum"],
        "AMT_PAYMENT": ["min", "max", "mean", "sum"],
        "DAYS_ENTRY_PAYMENT": ["max", "mean", "sum"],
    }
    result = aggregate(ins, spec, "INSTAL_")
    result["INSTAL_COUNT"] = ins.groupby(KEY).size().astype(np.float32)
    return result


def process_credit_card(raw_dir: Path) -> pd.DataFrame:
    """credit_card_balance : agrégats min/max/mean/sum/var de toutes les variables."""
    cc = read_csv(raw_dir / "credit_card_balance.csv", dtype={"NAME_CONTRACT_STATUS": "category"})
    cc = cc.drop(columns="SK_ID_PREV")
    cc, _ = one_hot_encode(cc, nan_as_category=True)
    stats = ["min", "max", "mean", "sum", "var"]
    spec = {c: stats for c in cc.columns if c != KEY}
    result = aggregate(cc, spec, "CC_")
    result["CC_COUNT"] = cc.groupby(KEY).size().astype(np.float32)
    return result


PROCESSORS: dict[str, Callable[[Path], pd.DataFrame]] = {
    "application": process_application,
    "bureau": process_bureau,
    "previous": process_previous,
    "pos": process_pos,
    "instalments": process_instalments,
    "credit_card": process_credit_card,
}


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_step(name: str, raw_dir: Path, out_dir: Path, force: bool = False) -> Path:
    """Exécute une étape, sauvegarde le résultat en parquet et libère la mémoire."""
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{name}.parquet"
    if target.exists() and not force:
        logger.info("[%s] déjà calculé -> ignoré (force=True pour recalculer)", name)
        return target
    start = time.time()
    df = PROCESSORS[name](Path(raw_dir))
    df.to_parquet(target)
    logger.info("[%s] %s en %.0f s -> %s", name, df.shape, time.time() - start, target.name)
    del df
    gc.collect()
    return target


def run_pipeline(raw_dir: Path, out_dir: Path, force: bool = False) -> None:
    """Exécute toutes les étapes dans l'ordre."""
    for name in PROCESSORS:
        run_step(name, raw_dir, out_dir, force)


def _parquet_columns(path: Path) -> list[str]:
    """Colonnes d'un parquet (sans l'index SK_ID_CURR), lues dans le schéma seul."""
    return [c for c in pq.read_schema(path).names if c not in (KEY, "__index_level_0__")]


def assemble_features(out_dir: Path) -> dict[str, int]:
    """Assemble les 6 tables en ``features_train.parquet`` et ``features_test.parquet``.

    La matrice finale est préallouée en float32 (pas de jointures pandas, qui
    dupliquent les données en mémoire). ``features_train`` contient la colonne TARGET ;
    ``features_test`` ne la contient pas.
    """
    out_dir = Path(out_dir)
    paths = {name: out_dir / f"{name}.parquet" for name in PROCESSORS}
    app_cols = _parquet_columns(paths["application"])  # contient TARGET
    columns = list(app_cols)
    for name in list(PROCESSORS)[1:]:
        columns += _parquet_columns(paths[name])
    if len(set(columns)) != len(columns):
        dupes = pd.Series(columns)[pd.Series(columns).duplicated()].unique()[:10]
        raise ValueError(f"Noms de colonnes dupliqués : {list(dupes)}")

    index = pd.read_parquet(paths["application"], columns=["TARGET"]).index
    n_rows = len(index)
    matrix = np.empty((n_rows, len(columns)), dtype=np.float32)
    start = 0
    for name, path in paths.items():
        table = pd.read_parquet(path)
        if name != "application":
            table = table.reindex(index)
        block = slice(start, start + table.shape[1])
        matrix[:, block] = table[_parquet_columns(path)].to_numpy(dtype=np.float32)
        start += table.shape[1]
        logger.info("[%s] %d colonnes ajoutées", name, table.shape[1])
        del table
        gc.collect()

    features = pd.DataFrame(matrix, index=index, columns=columns, copy=False)
    n_train = int(features["TARGET"].notna().sum())
    if features["TARGET"].iloc[:n_train].isna().any():
        raise ValueError("Les lignes train ne sont pas au début du fichier.")

    features.iloc[:n_train].to_parquet(out_dir / "features_train.parquet")
    test = features.iloc[n_train:].drop(columns="TARGET")
    test.to_parquet(out_dir / "features_test.parquet")
    logger.info("features_train %s | features_test %s", (n_train, len(columns)), test.shape)
    return {"n_train": n_train, "n_test": n_rows - n_train, "n_features": len(columns) - 1}
