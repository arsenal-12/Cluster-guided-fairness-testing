"""
experiments.py
==============

This script runs the full fairness testing pipeline on ADULT and COMPAS datasets.

We compare:
1. Random Search (baseline)
2. Cluster-Guided Search (proposed)
3. Adaptive Cluster-Guided Search (innovation)

The goal is to evaluate how effectively each method detects discriminatory behaviour.

All results are saved in the results/ folder.
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy import stats

warnings.filterwarnings("ignore")
os.makedirs("results", exist_ok=True)

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

ADULT_CFG = {
    "path": "dataset/processed_adult.csv",
    "model": "DNN/model_processed_adult.h5",
    "scaler": "DNN/scaler_adult.pkl",
    "target": "Class-label",
    "sensitive": ["age", "race", "gender"],
}

COMPAS_CFG = {
    "path": "dataset/processed_compas.csv",
    "model": "DNN/model_processed_compas.h5",
    "scaler": "DNN/scaler_compas.pkl",
    "target": "Recidivism",
    "sensitive": ["Race", "Sex", "Age"],
}

DISC_THRESHOLD = 0.20
RANDOM_STATE = 42
BATCH_SIZE = 256
N_RUNS = 5
K_DEFAULT = 20
BUDGET_DEFAULT = 5000
PILOT_DEFAULT = 1000

# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────

def load_dataset(cfg):
    """
    Loads dataset and returns only the TEST split.
    We intentionally test on unseen data to simulate real-world behaviour.
    """
    df = pd.read_csv(cfg["path"])
    X = df.drop(columns=[cfg["target"]])
    y = df[cfg["target"]]

    _, X_test, _, _ = train_test_split(
        X, y, test_size=0.3,
        random_state=RANDOM_STATE, stratify=y
    )

    return X_test.reset_index(drop=True)


def load_model_scaler(cfg):
    """
    Loads trained neural network and scaler.
    We treat the model as a BLACK BOX.
    """
    from tensorflow import keras
    model = keras.models.load_model(cfg["model"])
    scaler = joblib.load(cfg["scaler"])
    return model, scaler


# ─────────────────────────────────────────────
# CORE LOGIC
# ─────────────────────────────────────────────

def get_non_sensitive(X, sensitive):
    """Return features that are NOT sensitive."""
    return [c for c in X.columns if c not in sensitive]


def generate_batch_pairs(X, indices, sensitive, non_sensitive):
    """
    Generate counterfactual pairs:
    - A: original sample
    - B: same sample but sensitive features changed

    This simulates fairness testing (what if only gender/race changed?)
    """
    A = X.iloc[indices].values.copy().astype(float)
    B = A.copy()
    cols = list(X.columns)

    # Flip sensitive attributes
    for col in sensitive:
        if col in cols:
            ci = cols.index(col)
            unique_vals = X[col].unique()

            for i in range(len(indices)):
                current = B[i, ci]
                options = [v for v in unique_vals if v != current]
                if options:
                    B[i, ci] = np.random.choice(options)

    return A, B


def count_idi(model, scaler, A, B):
    """
    Count discriminatory pairs:
    If prediction difference > threshold → discrimination found
    """
    # Safety: skip empty batches
    if len(A) == 0 or len(B) == 0:
        return 0
    A = scaler.transform(A)
    B = scaler.transform(B)

    pa = model.predict(A, verbose=0).flatten()
    pb = model.predict(B, verbose=0).flatten()

    return np.sum(np.abs(pa - pb) > DISC_THRESHOLD)


# ─────────────────────────────────────────────
# METHODS
# ─────────────────────────────────────────────

def run_random(X, model, scaler, budget, sensitive):
    """
    Baseline method:
    Randomly sample inputs without any intelligence.
    """
    ns = get_non_sensitive(X, sensitive)

    found = 0
    done = 0

    while done < budget:
        bs = min(BATCH_SIZE, budget - done)
        idx = np.random.randint(0, len(X), bs)

        A, B = generate_batch_pairs(X, idx, sensitive, ns)
        found += count_idi(model, scaler, A, B)
        done += bs

    return found / budget


def run_cluster(X, model, scaler, budget, pilot, sensitive, k):
    """
    Proposed method:
    Use clustering to focus search on important regions.
    """
    ns = get_non_sensitive(X, sensitive
    X_scaled = StandardScaler().fit_transform(X[ns])
    labels = KMeans(n_clusters=k, random_state=RANDOM_STATE).fit_predict(X_scaled)

    # Pilot phase → estimate which clusters are promising
    scores = {}
    for c in range(k):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            scores[c] = 0
            continue

        chosen = np.random.choice(idx, size=min(50, len(idx)), replace=False)
        A, B = generate_batch_pairs(X, chosen, sensitive, ns)
        scores[c] = count_idi(model, scaler, A, B) / len(chosen)

    # Allocate budget based on scores
    total = sum(scores.values()) + 1e-9
    alloc = {c: int(budget * scores[c] / total) for c in scores}

    found = 0
    for c, n in alloc.items():
        idx = np.where(labels == c)[0]
        # Skip empty clusters or zero allocation
        if len(idx) == 0 or n == 0:
            continue
        chosen = np.random.choice(idx, size=n, replace=True)
        # Safety check (VERY important)
        if len(chosen) == 0:
            continue
        A, B = generate_batch_pairs(X, chosen, sensitive, ns)
        # Another safety check
        if len(A) == 0:
            continue
        found += count_idi(model, scaler, A, B)

    return found / budget


def run_adaptive_cluster(X, model, scaler, budget, pilot, sensitive):
    """
    INNOVATION:
    Instead of fixed K, we ADAPT the number of clusters.

    Idea:
    - Start simple (K=5)
    - Increase K only if needed
    """
    K = 5
    MAX_K = 25

    ns = get_non_sensitive(X, sensitive)

    while True:
        X_scaled = StandardScaler().fit_transform(X[ns])
        labels = KMeans(n_clusters=K, random_state=RANDOM_STATE).fit_predict(X_scaled)

        scores = []
        for c in range(K):
            idx = np.where(labels == c)[0]
            if len(idx) < 10:
                continue

            chosen = np.random.choice(idx, size=min(50, len(idx)), replace=False)
            A, B = generate_batch_pairs(X, chosen, sensitive, ns)

            score = count_idi(model, scaler, A, B) / len(chosen)
            scores.append(score)

        if len(scores) == 0:
            break

        # If clusters are too "mixed", increase K
        if np.var(scores) > 0.002 and K < MAX_K:
            K += 5
        else:
            break

    return run_cluster(X, model, scaler, budget, pilot, sensitive, K)


# ─────────────────────────────────────────────
# EXPERIMENTS
# ─────────────────────────────────────────────

def exp_main(X, model, scaler, sensitive, label):
    print(f"\n=== MAIN ({label}) ===")

    r = [run_random(X, model, scaler, BUDGET_DEFAULT, sensitive) for _ in range(N_RUNS)]
    c = [run_cluster(X, model, scaler, BUDGET_DEFAULT, PILOT_DEFAULT, sensitive, K_DEFAULT) for _ in range(N_RUNS)]

    print(f"Random:  {np.mean(r):.4f}")
    print(f"Cluster: {np.mean(c):.4f}")


def exp_adaptive(X, model, scaler, sensitive):
    print("\n=== ADAPTIVE vs FIXED ===")

    fixed = [run_cluster(X, model, scaler, BUDGET_DEFAULT, PILOT_DEFAULT, sensitive, K_DEFAULT) for _ in range(N_RUNS)]
    adaptive = [run_adaptive_cluster(X, model, scaler, BUDGET_DEFAULT, PILOT_DEFAULT, sensitive) for _ in range(N_RUNS)]

    print(f"Fixed:    {np.mean(fixed):.4f}")
    print(f"Adaptive: {np.mean(adaptive):.4f}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("\nRunning Fairness Experiments...\n")

    # Load datasets
    X_adult = load_dataset(ADULT_CFG)
    model_a, sc_a = load_model_scaler(ADULT_CFG)

    X_compas = load_dataset(COMPAS_CFG)
    model_c, sc_c = load_model_scaler(COMPAS_CFG)

    # Run core experiments
    exp_main(X_adult, model_a, sc_a, ADULT_CFG["sensitive"], "ADULT")
    exp_main(X_compas, model_c, sc_c, COMPAS_CFG["sensitive"], "COMPAS")

    # Innovation experiment
    exp_adaptive(X_adult, model_a, sc_a, ADULT_CFG["sensitive"])

    print("\nDone.\n")


if __name__ == "__main__":
    main()