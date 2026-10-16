"""Libellés français et formats d'affichage pour des utilisateurs non experts."""

from __future__ import annotations

import math

LABELS: dict[str, str] = {
    "AGE": "Âge (ans)",
    "EXT_SOURCE_1": "Score externe n°1 (0 à 1, plus haut = plus fiable)",
    "EXT_SOURCE_2": "Score externe n°2 (0 à 1, plus haut = plus fiable)",
    "EXT_SOURCE_3": "Score externe n°3 (0 à 1, plus haut = plus fiable)",
    "AMT_INCOME_TOTAL": "Revenu annuel",
    "AMT_CREDIT": "Montant du crédit demandé",
    "AMT_ANNUITY": "Annuité (remboursement annuel)",
    "AMT_GOODS_PRICE": "Prix du bien financé",
    "PAYMENT_RATE": "Poids de l'annuité dans le crédit (annuité / crédit)",
    "ANNUITY_INCOME_PERC": "Poids de l'annuité dans le revenu (annuité / revenu)",
    "INCOME_CREDIT_PERC": "Revenu rapporté au crédit (revenu / crédit)",
    "INCOME_PER_PERSON": "Revenu par personne du foyer",
    "DAYS_BIRTH": "Âge",
    "DAYS_EMPLOYED": "Ancienneté dans l'emploi actuel",
    "DAYS_EMPLOYED_PERC": "Ancienneté dans l'emploi rapportée à l'âge",
    "CNT_CHILDREN": "Nombre d'enfants",
    "CNT_FAM_MEMBERS": "Taille du foyer (personnes)",
    "APPROVED_CNT_PAYMENT_MEAN": "Durée moyenne (mois) des crédits précédents acceptés",
}

PREFIXES: dict[str, str] = {
    "BURO_": "Autres crédits (bureau de crédit)",
    "ACTIVE_": "Autres crédits en cours",
    "CLOSED_": "Autres crédits clôturés",
    "PREV_": "Demandes précédentes",
    "APPROVED_": "Demandes précédentes acceptées",
    "REFUSED_": "Demandes précédentes refusées",
    "POS_": "Crédits en point de vente",
    "INSTAL_": "Échéances de remboursement",
    "CC_": "Carte de crédit",
}

STATS: dict[str, str] = {
    "_MEAN": "moyenne",
    "_MAX": "maximum",
    "_MIN": "minimum",
    "_SUM": "total",
    "_VAR": "variabilité",
}


WORDS: dict[str, str] = {
    "DAYS": "jours", "CREDIT": "crédit", "AMT": "montant", "PAYMENT": "paiement",
    "INSTALMENT": "échéance", "ANNUITY": "annuité", "BALANCE": "solde", "MONTHS": "mois",
    "ENDDATE": "date de fin", "UPDATE": "mise à jour", "OVERDUE": "retard", "DEBT": "dette",
    "LIMIT": "limite", "SUM": "somme", "PERC": "ratio", "RATE": "taux", "DOWN": "acompte",
    "APPLICATION": "demande", "GOODS": "bien", "PRICE": "prix", "DRAWINGS": "retraits",
    "ATM": "distributeur", "CURRENT": "en cours", "CNT": "nombre", "DPD": "jours de retard",
    "DBD": "jours d'avance", "STATUS": "statut", "ENTRY": "effectué", "COUNT": "nombre",
    "VERSION": "version", "NUM": "nombre", "SIZE": "taille", "PROLONG": "prolongations",
    "DECISION": "décision", "HOUR": "heure", "PROCESS": "traitement", "START": "début",
}


def _translate(text: str) -> str:
    """Traduit mot à mot (les mots inconnus restent en minuscules)."""
    return " ".join(WORDS.get(word.upper(), word.lower()) for word in text.split("_") if word)


def label(name: str) -> str:
    """Libellé lisible d'une variable du modèle."""
    if name in LABELS:
        return LABELS[name]
    for prefix, text in PREFIXES.items():
        if name.startswith(prefix):
            rest = name[len(prefix):]
            stat = ""
            for suffix, word in STATS.items():
                if rest.endswith(suffix):
                    rest, stat = rest[: -len(suffix)], f" ({word})"
                    break
            return f"{text} – {_translate(rest)}{stat}"
    return name.replace("_", " ").lower().capitalize()


def _french_number(value: float, decimals: int) -> str:
    """Formate un nombre à la française : espace pour les milliers, virgule décimale."""
    text = f"{value:,.{decimals}f}"
    return text.replace(",", " ").replace(".", ",")


def format_value(name: str, value: float | None) -> str:
    """Valeur formatée pour l'affichage (None / NaN -> « non renseigné »)."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "non renseigné"
    if name in ("DAYS_BIRTH", "DAYS_EMPLOYED"):
        return f"{_french_number(abs(value) / 365.25, 1)} ans"
    if name == "AGE":
        return f"{_french_number(value, 0)} ans"
    if "DAYS" in name:
        return f"{_french_number(value, 0)} jours"
    if name.startswith("AMT_") or "_AMT_" in name:
        return _french_number(value, 0)
    if float(value).is_integer():
        return _french_number(value, 0)
    return _french_number(value, 3 if abs(value) < 10 else 1)
