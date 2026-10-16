# Scoring crédit – « Prêt à dépenser »

![CI](https://github.com/HAKASSOU/projet6-scoring-credit/actions/workflows/ci.yml/badge.svg)

Modèle de **scoring de défaut de paiement**, **API de prédiction** (FastAPI) et **dashboard
interactif** (Streamlit) destinés aux chargés de relation client d'une société de crédit à la
consommation. Projet 6 de la formation OpenClassrooms *Data Scientist*, réalisé sur le jeu
de données [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk).

## Résultats

| Indicateur | Valeur (jeu de test) |
|---|---|
| AUC | ≈ 0,79 |
| Seuil de décision (refus si probabilité ≥ seuil) | 0,09 |
| Part des demandes refusées | 27,5 % |
| Défauts détectés | 67,2 % |
| Bons clients refusés | 24,0 % |
| **Gain de coût vs « tout accepter »** | **39,9 %** |

Coût métier : un crédit accordé à un client qui fera défaut (FN) coûte **10 fois** plus qu'un
bon client refusé (FP). Le seuil de 0,09 trouvé empiriquement est quasi identique au seuil
théorique `c_FP / (c_FP + c_FN) = 1/11 ≈ 0,091`, ce qui confirme que les probabilités sont
bien calibrées.

## Architecture

```
CSV Home Credit ─► preprocessing.py ─► features_*.parquet ─► LightGBM ─► models/
   (8 tables)       (agrégats client)      (794 variables)               lgbm_model.txt
                                                                         metadata.json
                                                                              │
                          Streamlit (dashboard) ──── HTTP ────► FastAPI ◄─────┘
                          profil + comparaison          /predict : score, décision, facteurs
```

## Structure du dépôt

| Chemin | Rôle |
|---|---|
| `preprocessing.py` | Pipeline de préparation (adapté du kernel *LightGBM with Simple Features*), sobre en mémoire (`float32`, lecture par blocs, parquet) |
| `P6_01_notebook_preprocessing.ipynb` | Exploration de la cible et exécution du pipeline |
| `P6_02_notebook_modelisation.ipynb` | Entraînement, coût métier, choix du seuil, sauvegarde du modèle |
| `P6_03_notebook_interpretabilite.ipynb` | Interprétabilité SHAP (globale et locale) |
| `api/main.py` | API FastAPI : `/health`, `/clients`, `/predict` |
| `dashboard/` | Dashboard Streamlit (`app.py`, `charts.py`, `labels.py`) |
| `scripts/export_demo_data.py` | Extrait 2 000 clients de démonstration |
| `models/` | Modèle LightGBM, métadonnées, importances |
| `tests/` | Tests unitaires (API, dashboard) |
| `render.yaml`, `Dockerfile` | Déploiement de l'API |
| `.github/workflows/ci.yml` | Intégration continue (PEP8 + tests) |

## Méthodologie

- **Données** : 8 tables agrégées par client (`SK_ID_CURR`) → 307 507 clients d'entraînement,
  48 744 clients de test. Cible déséquilibrée (≈ 8 % de défauts).
- **Modèle** : LightGBM, découpage stratifié train / validation / test (70 / 15 / 15),
  *early stopping* sur l'AUC de validation. Pas de `is_unbalance` afin de conserver des
  **probabilités calibrées**, affichées telles quelles aux utilisateurs.
- **Décision** : seuil choisi sur la validation en minimisant le coût métier, évalué sur
  le test (jamais utilisé pour choisir).
- **Interprétabilité** : valeurs SHAP (globales) ; contributions par client calculées par
  LightGBM (`pred_contrib`) dans l'API et affichées en langage simple.
- **Éthique** : la variable `CODE_GENDER` est **exclue** du modèle (discrimination
  interdite dans la décision de crédit). Perte de performance négligeable (gain de coût
  de 40,2 % à 39,9 %).

## Installation et lancement (Windows)

```bat
pip install -r requirements-dev.txt

:: 1. Préparation des données et entraînement : exécuter les notebooks 01, 02, 03 dans l'ordre.
:: 2. Échantillon de démonstration
python scripts\export_demo_data.py

:: 3. API (terminal 1)
uvicorn api.main:app

:: 4. Dashboard (terminal 2)
streamlit run dashboard\app.py
```

L'API expose une documentation interactive sur <http://127.0.0.1:8000/docs>.
Le dashboard s'ouvre sur <http://localhost:8501>.

Exemple d'appel :

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" \
     -d "{\"sk_id_curr\": 100001}"
```

Variables d'environnement : `MODEL_DIR` et `CLIENTS_PATH` (API), `API_URL` et
`CLIENTS_PATH` (dashboard), `P6_DATA_DIR` (dossier des CSV pour les notebooks).

## Tests

```bat
python -m pytest -q tests
python -m pycodestyle --max-line-length=100 api dashboard scripts tests preprocessing.py
```

## Déploiement

- **API** : Render (*New > Blueprint*, dépôt GitHub, fichier `render.yaml`).
- **Dashboard** : Streamlit Community Cloud, fichier principal `dashboard/app.py`, avec le
  secret `API_URL = "https://<votre-service>.onrender.com"`.
- Sur l'offre gratuite de Render, l'API se met en veille : la première requête peut
  prendre environ une minute.

## Accessibilité du dashboard

Palette adaptée au daltonisme (bleu / orange), information jamais portée par la couleur
seule (symboles ✔ ✖ ▲ ▼ et textes), grandes polices, vocabulaire non technique, titres
hiérarchisés.

## Limites et pistes d'amélioration

- Le modèle repose sur des données historiques d'un seul opérateur : un suivi de la dérive
  (*data drift*) et un ré-entraînement périodique seraient nécessaires en production.
- Le coût métier (10 / 1) est une hypothèse à valider avec les équipes finance et risque.
- L'âge (`DAYS_BIRTH`) reste une variable du modèle : à examiner avec la conformité.
- Optimisation fine des hyperparamètres (gain attendu faible, de l'ordre de 0,002 à 0,004 d'AUC).
- Le dépôt contient un échantillon de démonstration (2 000 clients du jeu de test), à usage
  pédagogique uniquement.

## Auteur

Antoine Hakassou – projet réalisé dans le cadre de la formation OpenClassrooms *Data Scientist*.
