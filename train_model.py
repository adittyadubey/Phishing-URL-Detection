"""Train the classifiers used by the Streamlit app.

Two model sets are trained on the same split:
  * "full"     - all 44 dataset features (identical setup to the Colab notebook).
                 Used whenever the page's HTML can be downloaded.
  * "url_only" - the 17 features readable from the URL text. Used as a fallback
                 when a page cannot be opened (offline site, blocked, not HTML).

Run:  python train_model.py
Also called automatically by app.py if the saved models are missing or were
saved by an incompatible scikit-learn version.
"""
from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from phishing_features import URL_FEATURES

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "PhiUSIIL_10000_rows_clean.csv"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"
MODEL_PATH = MODEL_DIR / "phishing_url_detection_models.pkl"   # {"full": {...}, "url_only": {...}}
METADATA_PATH = MODEL_DIR / "model_metadata.json"

HEADLINE_MODEL = "Random Forest"   # model used for the main verdict
TEST_SIZE = 0.30
RANDOM_STATE = 42
CV_FOLDS = 2


def build_models() -> dict:
    """The four pipelines used by the project (same as the Colab notebook)."""
    imp = lambda: SimpleImputer(strategy="median")
    return {
        "Logistic Regression": Pipeline([("imputer", imp()), ("scaler", StandardScaler()),
                                         ("model", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE))]),
        "SVM (RBF)": Pipeline([("imputer", imp()), ("scaler", StandardScaler()),
                               ("model", SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE))]),
        "Decision Tree": Pipeline([("imputer", imp()), ("model", DecisionTreeClassifier(random_state=RANDOM_STATE))]),
        "Random Forest": Pipeline([("imputer", imp()), ("model", RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1))]),
    }


def _fit_and_score(X_train, X_test, y_train, y_test):
    """Tune on training folds only; evaluate once on the untouched test set."""
    actual_phish = (y_test == 0)
    fitted, scores = {}, {}
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    grids = {
        "Logistic Regression": {"model__C": [0.1, 1.0]},
        "SVM (RBF)": {"model__C": [1.0, 5.0], "model__gamma": ["scale"]},
        "Decision Tree": {"model__max_depth": [None, 24], "model__min_samples_leaf": [1]},
        "Random Forest": {"model__n_estimators": [150, 200], "model__max_depth": [None],
                          "model__min_samples_leaf": [1]},
    }
    for name, base_model in build_models().items():
        search = GridSearchCV(base_model, grids[name], scoring="f1", cv=cv, n_jobs=2,
                              refit=True, error_score="raise")
        search.fit(X_train, y_train)
        model = search.best_estimator_
        pred = model.predict(X_test)
        pred_phish = (pred == 0)
        col = list(model.classes_).index(0)
        scores[name] = {
            "accuracy": accuracy_score(actual_phish, pred_phish),
            "precision": precision_score(actual_phish, pred_phish, zero_division=0),
            "recall": recall_score(actual_phish, pred_phish, zero_division=0),
            "f1": f1_score(actual_phish, pred_phish, zero_division=0),
            "roc_auc": roc_auc_score(actual_phish, model.predict_proba(X_test)[:, col]),
            # Rows/columns: actual/predicted [legitimate, phishing].
            "confusion_matrix": confusion_matrix(actual_phish, pred_phish, labels=[False, True]).tolist(),
            "best_params": search.best_params_,
            "best_cv_f1": float(search.best_score_),
        }
        fitted[name] = model
    return fitted, scores


def train() -> dict:
    MODEL_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Cleaning identical to the notebook: drop missing labels, exact duplicates, constant columns.
    df = pd.read_csv(DATA_PATH).dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)
    if not set(df["label"].unique()).issubset({0, 1}):
        raise ValueError("Expected binary labels 0=phishing and 1=legitimate.")
    # Exact duplicate rows are removed before splitting. Raw URLs are not present in
    # this CSV, so URL/domain-level grouping cannot be performed reliably.
    before_dedup = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    duplicates_removed = before_dedup - len(df)
    X = df.drop(columns=["label"])
    X = X.drop(columns=[c for c in X.columns if X[c].nunique(dropna=False) <= 1])
    y = df["label"]
    full_features = list(X.columns)

    # Project convention: label 0 = phishing, label 1 = legitimate.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    full_models, full_scores = _fit_and_score(X_train, X_test, y_train, y_test)
    url_models, url_scores = _fit_and_score(X_train[URL_FEATURES], X_test[URL_FEATURES], y_train, y_test)

    # Build tld_freq imputation lookup from TRAINING ROWS ONLY to avoid test leakage.
    train_frame = X_train.copy()
    com_value = float(train_frame["tld_freq"].max())
    other = train_frame[train_frame["tld_freq"] < com_value]
    tld_lookup = {
        "com": com_value,
        "by_length": {str(k): float(v) for k, v in other.groupby("tld_len")["tld_freq"].median().items()},
        "default": float(other["tld_freq"].median()),
    }

    head = full_scores[HEADLINE_MODEL]
    metadata = {
        "selected_model": HEADLINE_MODEL,
        "dataset": "PhiUSIIL",
        "label_mapping": {"0": "phishing", "1": "legitimate"},
        "features": full_features,
        "url_features": URL_FEATURES,
        "n_rows": int(len(df)),
        "duplicates_removed": int(duplicates_removed),
        "raw_url_column_available": False,
        "domain_group_split": False,
        "validation": {"method": "StratifiedKFold", "folds": CV_FOLDS, "scoring": "f1", "tuning_data": "training split only"},
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "metrics": {k: head[k] for k in ("accuracy", "precision", "recall", "f1", "roc_auc")},
        "confusion_matrix": head["confusion_matrix"],
        "models": full_scores,
        "url_only_models": url_scores,
        "tld_lookup": tld_lookup,
        "label_definition": "0=phishing (positive class for precision/recall/F1), 1=legitimate",
    }
    joblib.dump({"full": full_models, "url_only": url_models}, MODEL_PATH, compress=3)
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    rf = full_models[HEADLINE_MODEL].named_steps["model"]
    pd.DataFrame({"feature": full_features, "importance": rf.feature_importances_}) \
        .sort_values("importance", ascending=False).to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)
    rows = []
    for tag, sc in (("All 44 features", full_scores), ("URL text only (17)", url_scores)):
        rows += [{"Feature set": tag, "Model": n, **{k: v for k, v in m.items() if k != "confusion_matrix"}} for n, m in sc.items()]
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    return metadata


if __name__ == "__main__":
    meta = train()
    print(f"{meta['n_test']} held-out URLs; phishing = positive class")
    for title, key in (("ALL 44 FEATURES (matches the Colab notebook)", "models"), ("URL TEXT ONLY (fallback)", "url_only_models")):
        print(f"\n{title}\n{'Model':24s} acc    prec   recall f1     auc")
        for name, m in meta[key].items():
            print(f"{name:24s} {m['accuracy']:.4f} {m['precision']:.4f} {m['recall']:.4f} {m['f1']:.4f} {m['roc_auc']:.4f}")
    print("\ntld_lookup:", meta["tld_lookup"])
