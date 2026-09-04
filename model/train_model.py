"""
train_model.py
Trains a LightGBM multi-class classifier to predict JMA seismic intensity
(shindo) from waveform features, handling severe class imbalance.

Run this from inside the model/ folder:
    python train_model.py
It expects ../data/eew_dataset.csv to exist (built by data/build_dataset.py).
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.utils.class_weight import compute_sample_weight
import joblib

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_PATH = "../data/eew_dataset.csv"
MODEL_OUT = "eew_model.txt" # native LightGBM format (easy to load in api/)
SKLEARN_MODEL_OUT = "eew_model.pkl" # sklearn-wrapper pickle (convenient too)
TARGET_COL = "shindo"
RANDOM_STATE = 42

# Must match the order used in build_dataset.py's SHINDO_LABELS
SHINDO_LABELS = ["0", "1", "2", "3", "4", "5-", "5+", "6-", "6+", "7"]


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Expected target column '{TARGET_COL}' not found in {path}")
    return df


def main():
    print(f"Loading dataset from {DATA_PATH} ...")
    df = load_data(DATA_PATH)
    print(f"Loaded {len(df)} rows, {df.shape[1]} columns")

    # Features = everything except the target
    X = df.drop(columns=[TARGET_COL, "true_intensity_value"])
    y_raw = df[TARGET_COL].astype(str)

    # Encode labels to integers 0..N-1 in a fixed, meaningful order
    label_to_idx = {label: i for i, label in enumerate(SHINDO_LABELS)}
    missing = set(y_raw.unique()) - set(label_to_idx)
    if missing:
        raise ValueError(f"Found labels not in SHINDO_LABELS: {missing}")
    y = y_raw.map(label_to_idx)

    print("\nClass distribution:")
    print(y_raw.value_counts().reindex(SHINDO_LABELS, fill_value=0))

    # Stratified split so rare classes appear in both train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"\nTrain: {len(X_train)} rows | Test: {len(X_test)} rows")

    # Sample weights to counter class imbalance (classes range ~48 to ~1360 rows)
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)

    # ------------------------------------------------------------------
    # Train LightGBM multi-class classifier
    # ------------------------------------------------------------------
    model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=len(SHINDO_LABELS),
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=10, # lower default since some classes are tiny
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    print("\nTraining LightGBM model...")
    model.fit(
        X_train, y_train,
        sample_weight=sample_weight,
        eval_set=[(X_test, y_test)],
        eval_metric="multi_logloss",
        callbacks=[lgb.early_stopping(stopping_rounds=30), lgb.log_evaluation(period=50)],
    )

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    y_pred = model.predict(X_test)

    print("\n" + "=" * 60)
    print("Classification report")
    print("=" * 60)
    print(classification_report(
        y_test, y_pred,
        labels=list(range(len(SHINDO_LABELS))),
        target_names=SHINDO_LABELS,
        zero_division=0,
    ))

    macro_f1 = f1_score(y_test, y_pred, average="macro")
    print(f"Macro F1: {macro_f1:.4f}")

    print("\n" + "=" * 60)
    print("Confusion matrix (rows=true, cols=predicted)")
    print("=" * 60)
    cm = confusion_matrix(y_test, y_pred, labels=list(range(len(SHINDO_LABELS))))
    cm_df = pd.DataFrame(cm, index=SHINDO_LABELS, columns=SHINDO_LABELS)
    print(cm_df)

    # ------------------------------------------------------------------
    # Save model
    # ------------------------------------------------------------------
    model.booster_.save_model(MODEL_OUT)
    joblib.dump(model, SKLEARN_MODEL_OUT)
    print(f"\nSaved model -> {MODEL_OUT} (native LightGBM) and {SKLEARN_MODEL_OUT} (sklearn pickle)")
    print("Label order for decoding predictions:", SHINDO_LABELS)


if __name__ == "__main__":
    main()