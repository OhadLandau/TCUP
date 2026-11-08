"""
Standalone Monte Carlo Gene Importance Script
Loads the trained TCUP v2 model and runs the Monte Carlo gene ablation on the held-out test set without retraining
"""
import os

os.environ["OMP_NUM_THREADS"] = "1"

import sys
import glob
import json
import time
import types
import pickle

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm
from joblib import Parallel, delayed

SEED = 42
num_monte_carlo_cycles = 10
noise_std = 20.0
block_size = 140
n_workers = max(1, (os.cpu_count() or 2) - 1)
ALPHA = 0.05

# Paths
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir = os.getcwd()
project_dir = os.environ.get("TCUP_DATA", script_dir)
output_dir = os.environ.get("TCUP_OUTPUT", os.path.join(project_dir, "TCUP v2 Results"))
data_dir = os.path.join(output_dir, "01_data")
model_dir = os.environ.get("TCUP_MODEL", os.path.join(output_dir, "TCUP_model"))
mc_dir = os.path.join(output_dir, "04_montecarlo")
os.makedirs(os.path.join(mc_dir, "raw"), exist_ok=True)


# ---------- Model forward pass (NumPy weights exported from the trained networks) ----------
def dense(x, w, p, i):
    return x @ w[f"{p}_{i}_0"] + w[f"{p}_{i}_1"]


def bn(x, w, p, i):
    gamma, beta, mean, var = (w[f"{p}_{i}_{j}"] for j in range(4))
    return (x - mean) / np.sqrt(var + 1e-3) * gamma + beta


def relu(x):
    return np.maximum(x, 0)


def snn(w, x):
    for i in (0, 2, 4):
        x = bn(relu(dense(x, w, "snn", i)), w, "snn", i + 1)
    return x


def cae(w, x):
    x = bn(relu(dense(x, w, "cae", 0)), w, "cae", 1)
    x = bn(relu(dense(x, w, "cae", 2)), w, "cae", 3)
    x = np.tanh(dense(x, w, "cae", 4))
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def meta(w, x):
    for i in range(3):
        x = relu(dense(x, w, "meta", i))
    logits = dense(x, w, "meta", 3)
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def load_state(data_dir, model_dir):
    cache = sys.modules.setdefault("tcup_mc_cache", types.ModuleType("tcup_mc_cache"))
    if getattr(cache, "key", None) == (data_dir, model_dir):
        return cache
    samples = pd.read_csv(os.path.join(data_dir, "samples.csv"))
    X = np.load(os.path.join(data_dir, "X_scaled_features.npy"))
    with open(os.path.join(model_dir, "trained_base_classifiers.pkl"), "rb") as fh:
        clfs = pickle.load(fh)
    for name, c in clfs.items():
        if name == "XGBoost":
            c.set_params(n_jobs=1)
        elif hasattr(c, "n_jobs"):
            c.n_jobs = 1
    with open(os.path.join(model_dir, "label_encoder.pkl"), "rb") as fh:
        cache.n = len(pickle.load(fh).classes_)
    z = np.load(os.path.join(model_dir, "nets_numpy.npz"), allow_pickle=False)
    cache.nets = {k: z[k] for k in z.files}
    cache.Xte = X[(samples["SPLIT"] == "test").values]
    cache.clfs = clfs
    cache.key = (data_dir, model_dir)
    return cache


def get_predictions(state, X):
    emb = np.hstack([snn(state.nets, X), cae(state.nets, X)]).astype(np.float32)
    feats = []
    for c in state.clfs.values():
        res = np.zeros((len(X), state.n), dtype=np.float32)
        res[:, c.seen_classes_] = c.predict_proba(emb)
        feats.append(res)
    return meta(state.nets, np.hstack(feats)).argmax(1).astype(np.uint8)


def ablate(data_dir, model_dir, gene_idx, noise_seed, sigma):
    state = load_state(data_dir, model_dir)
    X = state.Xte.copy()
    if len(gene_idx):
        rng = np.random.default_rng(noise_seed)
        X[:, gene_idx] += rng.normal(0, sigma, size=(X.shape[0], len(gene_idx))).astype(np.float32)
    return get_predictions(state, X)


def bh(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    res = np.empty_like(q)
    res[order] = np.minimum(q, 1)
    return res


def against_all_genes(values):
    med = np.median(values)
    mad = 1.4826 * np.median(np.abs(values - med))
    z = (values - med) / (mad if mad > 0 else np.std(values) + 1e-12)
    p = norm.sf(z)
    return z, p, bh(p)


if __name__ == "__main__":
    with open(os.path.join(model_dir, "feature_sets.pkl"), "rb") as fh:
        fcols = pickle.load(fh)["FullLasso"]
    with open(os.path.join(model_dir, "label_encoder.pkl"), "rb") as fh:
        labels = list(pickle.load(fh).classes_)
    samples = pd.read_csv(os.path.join(data_dir, "samples.csv"))
    test = samples[samples["SPLIT"] == "test"].reset_index(drop=True)
    y_test_true = np.searchsorted(labels, test["LABEL"].values)
    print(f"Features: {len(fcols)}")
    print(f"Test samples: {len(test)}")

    pool = Parallel(n_jobs=n_workers, backend="loky")
    args = (data_dir, model_dir)

    print("Generating baseline predictions...")
    baseline = pool(delayed(ablate)(*args, [], 0, 0.0) for _ in range(1))[0]
    np.save(os.path.join(mc_dir, "baseline_predictions.npy"), baseline)
    print(f"Baseline test accuracy (no noise): {(baseline == y_test_true).mean():.4f}")

    print("\n" + "=" * 80)
    print("STARTING MONTE CARLO GENE IMPORTANCE ANALYSIS")
    print("=" * 80)
    genes = list(range(len(fcols)))
    for cycle_idx in range(num_monte_carlo_cycles):
        for start in range(0, len(genes), block_size):
            path = os.path.join(mc_dir, "raw", f"cycle{cycle_idx:02d}_block{start:05d}.npy")
            if os.path.exists(path):
                continue
            t = time.time()
            batch = genes[start:start + block_size]
            for attempt in range(5):
                try:
                    preds = pool(delayed(ablate)(*args, [g], [SEED, cycle_idx, g], noise_std) for g in batch)
                    break
                except Exception as exc:
                    print(f"  Block failed ({exc!r}), attempt {attempt + 1} of 5")
                    pool = Parallel(n_jobs=n_workers, backend="loky")
            else:
                raise RuntimeError("Block failed five times")
            np.save(path, np.vstack(preds))
            print(f"  Cycle {cycle_idx + 1}/{num_monte_carlo_cycles}, genes {start}-{start + len(batch)}: "
                  f"{(time.time() - t) / len(batch):.2f}s per gene", flush=True)

    # ---------- Average across cycles ----------
    files = sorted(glob.glob(os.path.join(mc_dir, "raw", "*.npy")))
    n_cycles = len({os.path.basename(f)[:7] for f in files})
    P = np.vstack([np.load(f) for f in files]).reshape(n_cycles, -1, len(test))
    correct_after_noise = (P == y_test_true[None, None, :]).mean(axis=0)
    drop = (baseline == y_test_true)[None, :].astype(np.float32) - correct_after_noise

    # each gene's drop against the drops of all genes (robust z), Benjamini-Hochberg corrected
    df_final = pd.DataFrame({"Gene": fcols})
    groups = [("", np.ones(len(test), bool)), ("_Metastatic", test["SOURCE"].values == "METASTATIC")]
    groups += [(f"_{lab}", test["LABEL"].values == lab) for lab in labels if (test["LABEL"].values == lab).any()]
    for suffix, mask in groups:
        d = drop[:, mask].mean(axis=1)
        z, p, q = against_all_genes(d)
        df_final[f"AccuracyDrop{suffix}"] = d
        if suffix == "":
            df_final["Z_score"], df_final["p_value"] = z, p
        df_final[f"FDR{suffix}"] = q
    df_final.sort_values("AccuracyDrop", ascending=False, inplace=True)
    final_csv_path = os.path.join(model_dir, "monte_carlo_gene_importance_averaged.csv")
    df_final.to_csv(final_csv_path, index=False)
    print(f"\nFinal aggregated CSV saved: {final_csv_path}")
    print(f"Genes significant overall (FDR < {ALPHA}): {int((df_final['FDR'] < ALPHA).sum())}")
    print(f"Genes significant for metastatic samples (FDR < {ALPHA}): {int((df_final['FDR_Metastatic'] < ALPHA).sum())}")

    # Plot top 20
    top_k = 20
    df_top_20 = df_final.head(top_k)
    plt.figure(figsize=(10, 6))
    plt.barh(df_top_20["Gene"][::-1], df_top_20["AccuracyDrop"][::-1], color="red", alpha=0.6)
    plt.xlabel("Average Drop in Accuracy (Over Monte Carlo Cycles)")
    plt.title(f"Top {top_k} Genes by Avg Accuracy Drop (Noise={noise_std}, n={n_cycles})")
    plt.tight_layout()
    plot_path = os.path.join(mc_dir, "monte_carlo_gene_importance_top20_aggregated.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Plot saved: {plot_path}")

    print("\n" + "=" * 80)
    print("MONTE CARLO COMPLETE!")
    print("=" * 80)
