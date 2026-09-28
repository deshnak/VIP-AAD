"""
EXAQC-style variational quantum classifier (VQC) baseline.

VIP Fall 2026, Quantum Optimization subteam. One script for any of the team's
UCI benchmark datasets, so the same seed circuit can be run per person's dataset.

Reproduces the setup the subteam standardized on for the proposal (slide 7):
    - RY angle encoding of the scaled features (one qubit per feature, PCA if wide)
    - RY + RZ trainable rotation on every qubit
    - CNOT ring entanglement
    - readout on ceil(log2 n_classes) qubits, first n_classes outcomes -> classes
    - cross-entropy loss, COBYLA (gradient-free) or Adam
    - 70/30 stratified split, multi-seed restarts, reporting mean / max / std
    - a classical logistic-regression reference on the same split

Reference: Kar, Krutz & Desell (2026), "Investigating Quantum Circuit Designs
Using Neuro-Evolution" (EXAQC). Table 1: Iris up to 90.0%, Seeds 90.5-95.2%.

Usage:
    python exaqc_baseline.py --dataset iris
    python exaqc_baseline.py --dataset seeds

Seeds is pulled from a public mirror of the UCI set on first run and cached
locally; the other datasets ship with scikit-learn.
"""

import argparse
import os
import urllib.request

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from scipy.optimize import minimize
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

MAX_QUBITS = 8          # EXAQC used 6-8 input qubits; PCA down to this if wider
EPS = 1e-9
SEEDS_URL = ("https://raw.githubusercontent.com/maskot1977/ipython_notebook/"
             "master/toydata/seeds_dataset.txt")
SEEDS_CACHE = "seeds_dataset.txt"


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------
def load_dataset(name):
    if name == "iris":
        d = load_iris();          return d.data, d.target
    if name == "wine":
        d = load_wine();          return d.data, d.target
    if name in ("breast_cancer", "cancer"):
        d = load_breast_cancer(); return d.data, d.target
    if name == "seeds":
        if not os.path.exists(SEEDS_CACHE):
            urllib.request.urlretrieve(SEEDS_URL, SEEDS_CACHE)
        raw = np.genfromtxt(SEEDS_CACHE)
        return raw[:, :-1], raw[:, -1].astype(int) - 1
    raise ValueError(f"unknown dataset: {name}")


def prepare(name, split_seed=42):
    X, y = load_dataset(name)
    n_classes = len(np.unique(y))
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=split_seed)
    if X_tr.shape[1] > MAX_QUBITS:                 # keep circuit simulable
        pca = PCA(n_components=MAX_QUBITS, random_state=split_seed).fit(X_tr)
        X_tr, X_te = pca.transform(X_tr), pca.transform(X_te)
    scaler = MinMaxScaler(feature_range=(0, np.pi)).fit(X_tr)   # fit on train only
    return scaler.transform(X_tr), scaler.transform(X_te), y_tr, y_te, n_classes


def classical_baseline(name, split_seed=42):
    X, y = load_dataset(name)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=split_seed)
    sc = StandardScaler().fit(X_tr)
    return LogisticRegression(max_iter=2000).fit(sc.transform(X_tr), y_tr).score(
        sc.transform(X_te), y_te)


# ======================================================================
# EVOLUTION KNOBS --- everything LLM-GE is allowed to mutate lives here.
# Baseline is deliberately minimal so evolution has room to improve it.
# ======================================================================
def ansatz(params, n_qubits, n_layers):
    """RY+RZ per qubit, then a CNOT ring. params shape: (n_layers, n_qubits, 2)."""
    for layer in range(n_layers):
        for q in range(n_qubits):
            qml.RY(params[layer, q, 0], wires=q)
            qml.RZ(params[layer, q, 1], wires=q)
        for q in range(n_qubits):                 # entanglement pattern
            qml.CNOT(wires=[q, (q + 1) % n_qubits])
# ======================================================================


def make_circuit(n_qubits, n_readout):
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit(x, params, n_layers):
        qml.AngleEmbedding(x, wires=range(n_qubits), rotation="Y")
        ansatz(params, n_qubits, n_layers)
        return qml.probs(wires=list(range(n_readout)))
    return circuit


def make_helpers(circuit, n_qubits, n_classes):
    def class_probs(x, params, n_layers):
        p = circuit(x, params, n_layers)[..., :n_classes]
        return p / (p.sum(axis=-1, keepdims=True) + EPS)

    def cost(pf, X, y, n_layers, lib=np):
        p = class_probs(X, pf.reshape(n_layers, n_qubits, 2), n_layers)
        return -lib.mean(lib.log(p[np.arange(len(y)), y] + EPS))

    def accuracy(pf, X, y, n_layers):
        p = class_probs(X, np.asarray(pf).reshape(n_layers, n_qubits, 2), n_layers)
        return float(np.mean(np.argmax(p, axis=-1) == y))
    return cost, accuracy


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------
def train_cobyla(cost, n_qubits, X, y, n_layers, seed, maxiter):
    rng = np.random.default_rng(seed)
    init = rng.uniform(-np.pi, np.pi, size=n_layers * n_qubits * 2)
    return minimize(cost, init, args=(X, y, n_layers),
                    method="COBYLA", options={"maxiter": maxiter}).x


def train_adam(cost, n_qubits, X, y, n_layers, seed, steps=150, lr=0.05):
    rng = np.random.default_rng(seed)
    p = pnp.array(rng.uniform(-np.pi, np.pi, (n_layers, n_qubits, 2)), requires_grad=True)
    opt = qml.AdamOptimizer(lr)
    Xb = pnp.array(X)
    for _ in range(steps):
        p = opt.step(lambda p: cost(p.flatten(), Xb, y, n_layers, lib=pnp), p)
    return np.array(p).flatten()


# ----------------------------------------------------------------------
# Experiment
# ----------------------------------------------------------------------
def run(name, n_layers=2, n_seeds=10, maxiter=300, optimizer="cobyla"):
    X_tr, X_te, y_tr, y_te, n_classes = prepare(name)
    n_qubits = X_tr.shape[1]
    n_readout = max(1, int(np.ceil(np.log2(n_classes))))
    circuit = make_circuit(n_qubits, n_readout)
    cost, accuracy = make_helpers(circuit, n_qubits, n_classes)

    accs = []
    for s in range(n_seeds):
        if optimizer == "cobyla":
            w = train_cobyla(cost, n_qubits, X_tr, y_tr, n_layers, s, maxiter)
        else:
            w = train_adam(cost, n_qubits, X_tr, y_tr, n_layers, s)
        accs.append(accuracy(w, X_te, y_te, n_layers))
    accs = np.array(accs)
    print(f"  {name:13s} {optimizer:6s} L={n_layers} "
          f"[{n_qubits}q]: mean={accs.mean():.3f} max={accs.max():.3f} std={accs.std():.3f}")
    return accs


def report(name):
    print(f"\n=== {name.upper()} ===")
    print(f"  classical logistic regression: {classical_baseline(name):.3f}")
    for L in (1, 2, 3):
        run(name, n_layers=L, n_seeds=10, maxiter=300, optimizer="cobyla")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="all",
                    choices=["iris", "seeds", "wine", "breast_cancer", "all"])
    args = ap.parse_args()
    targets = ["iris", "seeds"] if args.dataset == "all" else [args.dataset]
    for t in targets:
        report(t)
