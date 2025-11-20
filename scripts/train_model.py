import os
import json
import logging
from pathlib import Path
from typing import List

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# -------------------------
# Logging
# -------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [train] %(levelname)s: %(message)s"
)
log = logging.getLogger("train")

# -------------------------
# Paths
# -------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------
# Configuration
# -------------------------
TARGET_COL = "Class/ASD"  # Target column
TEST_SIZE = 0.2
RANDOM_STATE = 42
NUMERIC_FEATURES: List[str] = None  # Auto-detect if None
MODELS_TO_TRAIN = {
    "RandomForest": RandomForestClassifier(n_estimators=200, max_depth=15, random_state=RANDOM_STATE, n_jobs=-1),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, max_depth=7, random_state=RANDOM_STATE),
}

# -------------------------
# Train single dataset
# -------------------------
def train_dataset(dataset_path: Path):
    log.info(f"Loading dataset: {dataset_path}")
    df = pd.read_csv(dataset_path, sep=",", engine="python", on_bad_lines="skip")
    df.columns = df.columns.str.strip().str.replace("\ufeff", "")
    df = df.dropna(how="all")

    numeric_features = NUMERIC_FEATURES
    if numeric_features is None:
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        if TARGET_COL in numeric_features:
            numeric_features.remove(TARGET_COL)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found")

    df = df.dropna(subset=numeric_features + [TARGET_COL])

    if df[TARGET_COL].dtype == object:
        df[TARGET_COL] = df[TARGET_COL].astype(str).str.strip().str.upper()
        df["label"] = df[TARGET_COL].map({"YES": 1, "NO": 0})
        if df["label"].isna().any():
            raise ValueError("Invalid target labels found")
    else:
        df["label"] = df[TARGET_COL].astype(int)

    X = df[numeric_features].astype(float)
    y = df["label"].astype(int)

    stratify_param = y if (y.nunique() > 1 and y.value_counts().min() > 1) else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=stratify_param
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}
    best_model = None
    best_name = None
    best_score = -1

    for name, clf in MODELS_TO_TRAIN.items():
        log.info(f"Training {name}...")
        clf.fit(X_train_scaled, y_train)
        preds = clf.predict(X_test_scaled)
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)

        results[name] = {"accuracy": float(acc), "precision": float(prec), "recall": float(rec), "f1": float(f1)}
        log.info(f"{name} metrics: acc={acc:.4f}, prec={prec:.4f}, rec={rec:.4f}, f1={f1:.4f}")

        if acc > best_score:
            best_score = acc
            best_model = clf
            best_name = name

    log.info(f"Best model: {best_name} (accuracy={best_score:.4f})")

    # Save artifacts
    joblib.dump(best_model, MODELS_DIR / "best_model.pkl")
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    with open(MODELS_DIR / "feature_names.json", "w") as f:
        json.dump(numeric_features, f, indent=2)
    with open(MODELS_DIR / "model_results.json", "w") as f:
        json.dump({
            "best_model": best_name,
            "best_accuracy": best_score,
            "model_results": results,
            "feature_names": numeric_features,
            "class_labels": {0: "NO", 1: "YES"}
        }, f, indent=2)

    log.info("✅ Training and saving complete.")

# -------------------------
# Entry point for multiple datasets
# -------------------------
if __name__ == "__main__":
    datasets = [
        PROJECT_ROOT / "Autism-Adult_Data.csv"
    ]
    for ds in datasets:
        train_dataset(ds)
