"""
train_model.py
==============
Trains a Deep Neural Network on either the ADULT or COMPAS dataset
and saves the model and scaler for use in fairness testing.

Usage:
    python train_model.py --dataset adult
    python train_model.py --dataset compas
    python train_model.py --dataset both   (trains both, recommended)

Outputs (per dataset):
    DNN/model_processed_adult.h5  +  DNN/scaler_adult.pkl
    DNN/model_processed_compas.h5 +  DNN/scaler_compas.pkl
"""

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import joblib
import warnings
warnings.filterwarnings("ignore")

os.makedirs("DNN", exist_ok=True)

# ─────────────────────────────────────────────
# DATASET CONFIGURATIONS
# ─────────────────────────────────────────────
CONFIGS = {
    "adult": {
        "path":       "dataset/processed_adult.csv",
        "target":     "Class-label",
        "model_out":  "DNN/model_processed_adult.h5",
        "scaler_out": "DNN/scaler_adult.pkl",
        "epochs":     50,
        "batch_size": 64,
        "layers":     [64, 32, 16],
        "dropout":    [0.3, 0.2],
        "label":      "ADULT Income Dataset",
    },
    "compas": {
        "path":       "dataset/processed_compas.csv",
        "target":     "Recidivism",
        "model_out":  "DNN/model_processed_compas.h5",
        "scaler_out": "DNN/scaler_compas.pkl",
        "epochs":     60,
        "batch_size": 32,
        "layers":     [64, 32, 16],
        "dropout":    [0.3, 0.2],
        "label":      "COMPAS Recidivism Dataset",
    }
}

RANDOM_STATE = 42
TEST_SIZE    = 0.3


# ─────────────────────────────────────────────
# TRAINING FUNCTION
# ─────────────────────────────────────────────
def train(dataset_name):
    cfg = CONFIGS[dataset_name]

    print(f"\n{'='*52}")
    print(f"  Training DNN — {cfg['label']}")
    print(f"{'='*52}")

    # Load
    df = pd.read_csv(cfg["path"])
    print(f"  Dataset shape: {df.shape}")
    print(f"  Target: {cfg['target']}")
    print(f"  Class distribution:\n{df[cfg['target']].value_counts().to_string()}")

    X = df.drop(columns=[cfg["target"]]).values
    y = df[cfg["target"]].values

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE,
        random_state=RANDOM_STATE, stratify=y
    )
    print(f"\n  Train: {len(X_train)} | Test: {len(X_test)}")

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # Build DNN
    n_feat = X_train.shape[1]
    model  = keras.Sequential()
    model.add(layers.Input(shape=(n_feat,)))
    for i, units in enumerate(cfg["layers"]):
        model.add(layers.Dense(units, activation="relu"))
        model.add(layers.BatchNormalization())
        if i < len(cfg["dropout"]):
            model.add(layers.Dropout(cfg["dropout"][i]))
    model.add(layers.Dense(1, activation="sigmoid"))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    model.summary()

    # Train
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=7,
        restore_best_weights=True
    )
    model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        callbacks=[early_stop],
        verbose=1
    )

    # Evaluate
    y_prob = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_prob >= 0.5).astype(int)
    acc    = accuracy_score(y_test, y_pred)

    print(f"\n  Test Accuracy: {acc:.4f}")
    print("\n  Classification Report:")
    labels = ["<=50K",">50K"] if dataset_name=="adult" else ["No recidivism","Recidivism"]
    print(classification_report(y_test, y_pred, target_names=labels))

    # Save
    model.save(cfg["model_out"])
    joblib.dump(scaler, cfg["scaler_out"])
    print(f"  Model  saved → {cfg['model_out']}")
    print(f"  Scaler saved → {cfg['scaler_out']}")

    return acc


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train DNN for fairness testing")
    parser.add_argument("--dataset", choices=["adult","compas","both"],
                        default="both",
                        help="Which dataset to train on (default: both)")
    args = parser.parse_args()

    datasets = ["adult","compas"] if args.dataset=="both" else [args.dataset]

    results = {}
    for ds in datasets:
        acc = train(ds)
        results[ds] = acc

    print(f"\n{'='*52}")
    print("  Training Summary")
    print(f"{'='*52}")
    for ds, acc in results.items():
        print(f"  {ds.upper():<10} Test accuracy: {acc:.4f}")
    print(f"{'='*52}\n")


if __name__ == "__main__":
    main()