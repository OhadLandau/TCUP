import os
import sys
import glob
import json
import time
import pickle
import random
import hashlib
import shutil
import logging
import unittest
import warnings
import subprocess
from contextlib import contextmanager
from datetime import datetime
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from joblib import Parallel, delayed
from imblearn.over_sampling import SMOTE
from sklearn.impute import KNNImputer
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, r2_score)
from sklearn.utils import shuffle
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

PHASE = sys.argv[2] if len(sys.argv) > 2 and sys.argv[1] == "--phase" else None
if PHASE in ("meta", "export"):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
if PHASE in ("embed", "meta", "export"):
    import tensorflow as tf
    from tensorflow.keras import layers, models, regularizers, optimizers, backend as K
    from tensorflow.keras.utils import Sequence

    tf.config.experimental.enable_op_determinism()
    for g in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(g, True)


@contextmanager
def timer(name):
    start_time = time.time()
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting: {name}", flush=True)
    try:
        yield
    finally:
        elapsed = time.time() - start_time
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Completed {name} in {elapsed:.2f}s ({elapsed / 60:.1f}m)", flush=True)


# ---------- Paths ------------
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir = os.getcwd()

project_dir = os.environ.get("TCUP_DATA", script_dir)
patient_data_file = os.path.join(project_dir, "BalancedTCGA_CancerTranscriptomics_withIDs.csv")
metastatic_data_file = os.path.join(project_dir, "metastatic_TOO_dataset_tpm.csv")
metastatic_clinical_file = os.path.join(project_dir, "data_clinical_sample.csv")
metastatic_raw_dir = os.path.join(project_dir, "RawReadsMetastaticDataset")
gtex_reads_dir = os.path.join(project_dir, "GTEx Reads")
output_dir = os.environ.get("TCUP_OUTPUT", os.path.join(project_dir, "TCUP v2 Results"))

# ---------- Settings ------------
SPLIT_SEED = 42
SEED_LIST = [42, 18, 309, 1992, 451]
MIN_GENES = 1500
LASSO_ALPHAS = (0.02, 0.01, 0.005, 0.0025, 0.00125, 0.001)
LASSO_REFINE_STEPS = 3
MET_TEST_COUNTS = {"LUAD": 20, "PAAD": 25, "BRCA": 40, "COAD": 35}
max_pairs = 1000000
max_usage_per_sample_posneg = 5
metastatic_pos_neg_count = 30
SNN_EPOCHS = (10, 10, 10, 10)
CAE_EPOCHS = (10, 10, 10, 30)
CLASSIFIERS = ["LogisticRegression", "KNN", "RandomForest", "SVM", "XGBoost"]
GRIDS = {
    "LogisticRegression": {"C": [0.1, 1.0, 10.0], "penalty": ["l1", "l2"], "solver": ["saga"], "max_iter": [500, 1000]},
    "KNN": {"n_neighbors": [3, 5, 7], "weights": ["uniform", "distance"], "metric": ["euclidean", "manhattan"], "p": [1, 2]},
    "RandomForest": {"n_estimators": [50, 100], "max_depth": [10, 20, None], "min_samples_split": [2, 5], "min_samples_leaf": [1, 2]},
    "SVM": {"C": [0.1, 1.0, 10.0], "kernel": ["rbf", "poly"], "gamma": ["scale", "auto"], "degree": [2, 3]},
    "XGBoost": {"n_estimators": [50, 100], "max_depth": [3, 5, 7], "learning_rate": [0.01, 0.1], "subsample": [0.8, 1.0]},
}
restricted_archs = [
    (128, 128, 128, 0.3, 1e-4, 'adam', 64, 20),
    (256, 64, 256, 0.3, 1e-5, 'adam', 32, 20),
    (256, 256, 64, 0.3, 1e-5, 'adam', 16, 20),
    (256, 256, 256, 0.3, 1e-5, 'adam', 32, 20),
    (256, 256, 256, 0.3, 1e-4, 'adam', 64, 20),
    (128, 128, 256, 0.3, 1e-5, 'adam', 32, 20),
    (256, 256, 128, 0.3, 1e-4, 'adam', 32, 20),
    (256, 128, 128, 0.3, 1e-5, 'adam', 64, 20),
]


def out(*parts):
    p = os.path.join(output_dir, *parts)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


data_dir = lambda *p: out("01_data", *p)
seed_dir = lambda seed, *p: out("02_embedders", f"seed_{seed}", *p)
variant_dir = lambda variant, seed, *p: out("03_stacking", variant, f"seed_{seed}", *p)
model_dir = lambda *p: out("TCUP_model", *p)
hyperparameters_txt_file = out("03_stacking", "base_classifiers_hyperparameters.txt")


# ---------- Data Processing ------------
def tpm(counts, lengths_kb):
    rate = counts / lengths_kb
    return rate / np.nansum(rate, axis=1, keepdims=True) * 1e6


def load_metastatic_gene_table():
    files = sorted(glob.glob(os.path.join(metastatic_raw_dir, "*.txt")))
    symbols = [c for c in pd.read_csv(metastatic_data_file, nrows=0, index_col=0).columns if c != "CancerType"]
    lengths, counts, raw_tpm = 0.0, {}, {}
    for f in files:
        df = pd.read_csv(f, sep="\t", usecols=["length", "expected_count", "TPM"])
        sid = os.path.basename(f).split("_")[0]
        lengths = lengths + df["length"].values
        counts[sid] = df["expected_count"].values.astype(np.float32)
        raw_tpm[sid] = df["TPM"].values.astype(np.float32)
    table = pd.DataFrame({"gene": [s.upper() for s in symbols], "mean_length": lengths / len(files)})
    return table, counts, raw_tpm, len(files)


def load_tcga():
    t = pd.read_csv(patient_data_file, index_col=0)
    t = t.reset_index().rename(columns={t.index.name or "index": "CANCER"})
    n_all = len(t)
    t = t[t["SampleID"] != "Entrez_Gene_Id"]
    n_annotation = n_all - len(t)
    t.columns = [c.upper() if c not in ("SampleID", "CANCER", "class") else c for c in t.columns]
    t = t.loc[:, ~t.columns.duplicated()]
    t = t.drop_duplicates(subset=["SampleID"])
    t["PATIENT_ID"] = t["SampleID"].str.split("-").str[:3].str.join("-")
    t["SAMPLE_TYPE_CODE"] = t["SampleID"].str.split("-").str[3].str[:2]
    return t, n_annotation


def load_gtex():
    frames, meta = [], []
    for f in sorted(glob.glob(os.path.join(gtex_reads_dir, "gene_reads_*.csv"))):
        tissue = os.path.basename(f).replace("gene_reads_", "").split("_")[0].capitalize()
        df = pd.read_csv(f, index_col=0)
        df.index = df.index.str.upper()
        df = df[~df.index.duplicated(keep="first")]
        frames.append(df.astype(np.float32))
        meta += [(c, "-".join(c.split("-")[:2]), f"{tissue}_GTEx") for c in df.columns]
    m = pd.DataFrame(meta, columns=["SampleID", "DONOR_ID", "LABEL"])
    keep = m["LABEL"].map(m["LABEL"].value_counts()) >= 30
    return frames, m, keep.values


def group_split(labels, groups, seed):
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(np.zeros(len(labels)), labels, groups))
    split = np.empty(len(labels), dtype=object)
    split[:] = "train"
    split[folds[0][1]] = "test"
    split[folds[1][1]] = "val"
    return split


def row_hashes(X):
    return [hashlib.md5(np.round(r, 4).tobytes()).hexdigest() for r in X]


def prepare_data():
    if os.path.exists(data_dir("data_report.json")):
        print("Prepared data checkpoint found. Skipping data preparation.")
        return
    report = {}

    # ---------- Metastatic (BCGSC) data and gene lengths ----------
    print("Reading metastatic raw RSEM files...")
    table, bc_counts, bc_rawtpm, n_files = load_metastatic_gene_table()
    report["bcgsc_raw_files"] = n_files
    table.to_csv(data_dir("gene_lengths_from_bcgsc_rsem.csv"), index=False)

    met = pd.read_csv(metastatic_data_file, index_col=0)
    met_ids = met.index.astype(str)
    published = met.drop(columns="CancerType").values.astype(np.float32)
    agree = [np.allclose(published[i], bc_rawtpm[s], atol=0.02) for i, s in enumerate(met_ids)]
    report["bcgsc_row_order_verified_samples"] = f"{int(sum(agree))} of {len(agree)}"
    if not all(agree):
        raise RuntimeError("Metastatic gene order differs between the TPM CSV and raw files")
    met_lab = met["CancerType"].values
    published = None

    clin = pd.read_csv(metastatic_clinical_file)
    clin["SAMPLE_ID"] = clin["SAMPLE_ID"].astype(str)
    clin = clin.set_index("SAMPLE_ID").reindex(met_ids)
    same_site = clin["BIOPSY_SITE"].str.lower().values == clin["PRIMARY_SITE"].str.lower().values
    removed = pd.DataFrame({"SampleID": met_ids[same_site], "CancerType": met_lab[same_site],
                            "BIOPSY_SITE": clin["BIOPSY_SITE"].values[same_site],
                            "PRIMARY_SITE": clin["PRIMARY_SITE"].values[same_site]})
    removed.to_csv(data_dir("bcgsc_removed_non_discordant.csv"), index=False)
    report["bcgsc_input_samples"] = len(met_ids)
    report["bcgsc_removed_biopsy_equals_primary"] = int(same_site.sum())
    met_df = pd.DataFrame({"SampleID": met_ids[~same_site], "TISSUE": met_lab[~same_site]})
    vc = met_df["TISSUE"].value_counts()
    met_df = met_df[met_df["TISSUE"].map(vc) >= 5].reset_index(drop=True)
    report["bcgsc_after_class_filter"] = len(met_df)
    report["bcgsc_class_counts"] = met_df["TISSUE"].value_counts().to_dict()

    # ---------- TCGA and GTEx ----------
    print("Reading TCGA...")
    tcga, n_annotation = load_tcga()
    report["tcga_annotation_rows_removed"] = n_annotation
    report["tcga_samples"] = len(tcga)
    report["tcga_unique_patients"] = int(tcga["PATIENT_ID"].nunique())
    report["tcga_sample_type_codes"] = tcga["SAMPLE_TYPE_CODE"].value_counts().to_dict()
    report["tcga_sample_type_by_cancer_non_01"] = tcga[tcga["SAMPLE_TYPE_CODE"] != "01"]["CANCER"].value_counts().to_dict()

    print("Reading GTEx...")
    gtex_frames, gtex_meta, gtex_keep = load_gtex()
    report["gtex_samples_kept"] = int(gtex_keep.sum())
    report["gtex_donors"] = int(gtex_meta.loc[gtex_keep, "DONOR_ID"].nunique())

    meta_cols = {"SampleID", "CANCER", "class", "PATIENT_ID", "SAMPLE_TYPE_CODE"}
    tcga_genes = {c for c in tcga.columns if c not in meta_cols}
    bc_lengths = table.drop_duplicates("gene").set_index("gene")["mean_length"]
    genes = sorted(tcga_genes & set(gtex_frames[0].index) & set(bc_lengths[bc_lengths > 0].index))
    report["common_genes_with_length"] = len(genes)
    pd.Series(genes, name="gene").to_csv(data_dir("common_genes.csv"), index=False)
    lengths_kb = bc_lengths.loc[genes].values.astype(np.float64)[None, :] / 1000.0

    # ---------- TPM, same formula for all three sources ----------
    print(f"Computing TPM on {len(genes)} common genes...")
    pos = bc_lengths.index.get_indexer(genes)
    first = table.drop_duplicates("gene").index.values
    bc_matrix = np.vstack([bc_counts[s][first][pos] for s in met_df["SampleID"]]).astype(np.float64)
    bc_tpm = tpm(bc_matrix, lengths_kb)
    rho = [spearmanr(bc_tpm[i], bc_rawtpm[s][first][pos]).correlation for i, s in enumerate(met_df["SampleID"][:50])]
    report["bcgsc_formula_vs_rsem_tpm_spearman_mean_50_samples"] = float(np.mean(rho))
    bc_counts = bc_rawtpm = None

    tc_tpm = tpm(tcga[genes].values.astype(np.float64), lengths_kb)
    gt_mat = np.hstack([f.loc[genes].values for f in gtex_frames]).T[gtex_keep].astype(np.float64)
    gtex_frames = None
    gt_tpm = tpm(gt_mat, lengths_kb)
    gtex_meta = gtex_meta[gtex_keep].reset_index(drop=True)
    report["units_before_conversion_median_total"] = {
        "GTEx": float(np.median(gt_mat.sum(1))), "TCGA": float(np.median(np.nansum(tcga[genes].values, 1))),
        "BCGSC": float(np.median(bc_matrix.sum(1)))}
    gt_mat = bc_matrix = None

    X = np.log2(np.vstack([gt_tpm, tc_tpm, bc_tpm]) + 1).astype(np.float32)
    gt_tpm = tc_tpm = bc_tpm = None

    # ---------- Splits: GTEx by donor, TCGA by patient, metastatic held-out test ----------
    gtex_split = group_split(gtex_meta["LABEL"].values, gtex_meta["DONOR_ID"].values, SPLIT_SEED)
    tcga_labels = (tcga["CANCER"] + "_TCGA").values
    tcga_split = group_split(tcga_labels, tcga["PATIENT_ID"].values, SPLIT_SEED)

    rng = random.Random(SPLIT_SEED)
    test_idx = []
    for tissue, n in MET_TEST_COUNTS.items():
        pool = met_df.index[met_df["TISSUE"] == tissue].tolist()
        if len(pool) < n + 2:
            raise RuntimeError(f"{tissue}: {len(pool)} samples is too few for {n} test samples")
        test_idx += rng.sample(pool, n)
    met_split = np.empty(len(met_df), dtype=object)
    met_split[test_idx] = "test"
    dev = met_df.drop(index=test_idx)
    tr, va = train_test_split(dev.index.values, test_size=0.25, random_state=SPLIT_SEED, stratify=dev["TISSUE"])
    met_split[tr], met_split[va] = "train", "val"

    meta = pd.concat([
        pd.DataFrame({"SampleID": gtex_meta["SampleID"], "GROUP_ID": gtex_meta["DONOR_ID"], "LABEL": gtex_meta["LABEL"],
                      "SOURCE": "GTEX", "SPLIT": gtex_split}),
        pd.DataFrame({"SampleID": tcga["SampleID"].values, "GROUP_ID": tcga["PATIENT_ID"].values, "LABEL": tcga_labels,
                      "SOURCE": "TCGA", "SPLIT": tcga_split}),
        pd.DataFrame({"SampleID": met_df["SampleID"].values, "GROUP_ID": met_df["SampleID"].values,
                      "LABEL": met_df["TISSUE"].values + "_METASTATIC", "SOURCE": "METASTATIC", "SPLIT": met_split}),
    ], ignore_index=True)
    meta["SYNTHETIC"] = False
    tcga = None

    # ---------- KNN imputation, fitted on train ----------
    is_tr, is_va, is_te = (meta["SPLIT"].values == s for s in ("train", "val", "test"))
    report["missing_values_by_source"] = {s: int(np.isnan(X[meta["SOURCE"].values == s]).sum()) for s in ("GTEX", "TCGA", "METASTATIC")}
    with timer("KNN imputation"):
        imp = KNNImputer(n_neighbors=5)
        imp.fit(X[is_tr])
        for mask in (is_tr, is_va, is_te):
            rows = np.where(mask & np.isnan(X).any(axis=1))[0]
            if len(rows):
                X[rows] = imp.transform(X[rows]).astype(np.float32)
        imp = None
    if np.isnan(X).any():
        raise RuntimeError("Missing values remain after imputation")

    linear = np.power(2.0, X.astype(np.float64)) - 1
    linear = linear / linear.sum(axis=1, keepdims=True) * 1e6
    X = np.log2(linear + 1).astype(np.float32)
    units = []
    for s in ("GTEX", "TCGA", "METASTATIC"):
        mask = meta["SOURCE"].values == s
        units.append({"source": s, "samples": int(mask.sum()), "total_min": linear[mask].sum(1).min(), "total_max": linear[mask].sum(1).max(),
                      "median_log2": float(np.median(X[mask])), **{f"median_{g}": float(np.median(linear[mask, genes.index(g)])) for g in ("GAPDH", "ACTB", "EEF1A1", "ALB") if g in genes}})
    pd.DataFrame(units).to_csv(data_dir("units_by_source_after_tpm.csv"), index=False)
    linear = None

    # ---------- Leakage checks ----------
    hashes = np.array(row_hashes(X))
    checks = {}
    for a, b in (("train", "test"), ("val", "test"), ("train", "val")):
        ma, mb = meta["SPLIT"].values == a, meta["SPLIT"].values == b
        checks[f"expression_overlap_{a}_{b}"] = len(set(hashes[ma]) & set(hashes[mb]))
        checks[f"sampleid_overlap_{a}_{b}"] = len(set(meta["SampleID"][ma]) & set(meta["SampleID"][mb]))
        checks[f"donor_patient_overlap_{a}_{b}"] = len(set(meta["GROUP_ID"][ma]) & set(meta["GROUP_ID"][mb]))
    report["leakage_checks"] = checks
    report["duplicate_expression_rows_overall"] = int(len(hashes) - len(set(hashes)))
    if any(v != 0 for v in checks.values()):
        raise RuntimeError(f"Leakage check failed: {checks}")
    print("Leakage checks passed:", checks)

    # ---------- LASSO gene selection on real training samples ----------
    classes = sorted(meta.loc[is_tr, "LABEL"].unique())
    Xtr_std = StandardScaler().fit_transform(X[is_tr]).astype(np.float64)
    ytr = meta.loc[is_tr, "LABEL"].values
    path = []

    def fit_alpha(alpha):
        def one_vs_rest(label):
            return np.abs(Lasso(alpha=alpha, random_state=SPLIT_SEED, max_iter=3000, selection="random").fit(Xtr_std, (ytr == label).astype(float)).coef_)

        s = np.vstack(Parallel(n_jobs=4, prefer="threads")(delayed(one_vs_rest)(c) for c in classes)).max(axis=0)
        print(f"  LASSO alpha={alpha:.6f}: {int((s > 0).sum())} genes", flush=True)
        path.append({"alpha": alpha, "nonzero_genes": int((s > 0).sum())})
        return s

    with timer("Feature Selection (Lasso)"):
        weak, strong = None, None
        for alpha in LASSO_ALPHAS:
            s = fit_alpha(alpha)
            if (s > 0).sum() >= MIN_GENES:
                weak = (alpha, s)
                break
            strong = alpha
        if weak is None:
            raise RuntimeError(f"Even the weakest penalty keeps fewer than {MIN_GENES} genes")
        for _ in range(LASSO_REFINE_STEPS):
            if strong is None or (weak[1] > 0).sum() <= MIN_GENES * 1.1:
                break
            alpha = float(np.sqrt(weak[0] * strong))
            s = fit_alpha(alpha)
            if (s > 0).sum() >= MIN_GENES:
                weak = (alpha, s)
            else:
                strong = alpha
    chosen_alpha, score = weak
    Xtr_std = None
    features = [g for g, v in zip(genes, score) if v > 0]
    pd.DataFrame(path).to_csv(data_dir("lasso_alpha_path.csv"), index=False)
    pd.DataFrame({"gene": genes, "max_abs_coef": score, "selected": score > 0}).to_csv(data_dir("lasso_gene_scores.csv"), index=False)
    report["lasso_min_genes"] = MIN_GENES
    report["lasso_chosen_alpha"] = chosen_alpha
    report["lasso_alpha_path"] = path
    report["lasso_selected_genes"] = len(features)

    # ---------- Standard scaling, then SMOTE on training metastatic classes ----------
    idx = [genes.index(g) for g in features]
    scaler = StandardScaler().fit(X[is_tr][:, idx])
    XF = scaler.transform(X[:, idx]).astype(np.float32)
    rows = []
    syn_X, syn_y = [], []
    met_train = meta[(meta["SOURCE"] == "METASTATIC") & (meta["SPLIT"] == "train")]
    for lab, g in met_train.groupby("LABEL"):
        n = len(g)
        if n < 5:
            rows.append({"label": lab, "real": n, "synthetic": 0, "k": None})
            continue
        k = min(5, n - 1)
        others = XF[meta.index[(meta["SPLIT"] == "train") & (meta["SOURCE"] == "TCGA")][:k + 1]]
        Xs = np.vstack([XF[g.index], others])
        ys = np.array([1] * n + [0] * len(others))
        Xr, yr = SMOTE(sampling_strategy={1: int(np.ceil(n * 1.5))}, k_neighbors=k, random_state=SPLIT_SEED).fit_resample(Xs, ys)
        new = Xr[len(Xs):]
        syn_X.append(new)
        syn_y += [lab] * len(new)
        rows.append({"label": lab, "real": n, "synthetic": len(new), "k": k})
    pd.DataFrame(rows).to_csv(data_dir("smote_by_class.csv"), index=False)
    syn_X = np.vstack(syn_X).astype(np.float32)
    syn_meta = pd.DataFrame({"SampleID": [f"SMOTE_{i}" for i in range(len(syn_y))], "GROUP_ID": "SYNTHETIC", "LABEL": syn_y,
                             "SOURCE": "METASTATIC", "SPLIT": "train", "SYNTHETIC": True})
    all_meta = pd.concat([meta, syn_meta], ignore_index=True)
    all_X = np.vstack([XF, syn_X])

    h_syn = set(row_hashes(syn_X))
    h_hold = set(row_hashes(XF[(meta["SPLIT"] != "train").values]))
    report["synthetic_equal_to_holdout"] = len(h_syn & h_hold)
    h_tr = set(row_hashes(XF[is_tr]))
    h_te = set(row_hashes(XF[is_te]))
    report["feature_space_train_test_overlap"] = len(h_tr & h_te)
    if report["synthetic_equal_to_holdout"] or report["feature_space_train_test_overlap"]:
        raise RuntimeError("Leakage found after scaling or SMOTE")

    # ---------- Save ----------
    np.save(data_dir("X_log2tpm_all_genes.npy"), X)
    np.save(data_dir("X_scaled_features.npy"), all_X)
    all_meta.to_csv(data_dir("samples.csv"), index=False)
    pd.Series(features, name="gene").to_csv(data_dir("features.csv"), index=False)
    with open(data_dir("scaler.pkl"), "wb") as fh:
        pickle.dump(scaler, fh)
    labels = sorted(all_meta["LABEL"].unique())
    with open(data_dir("labels.json"), "w") as fh:
        json.dump(labels, fh, indent=1)
    counts = pd.crosstab(all_meta["LABEL"], [all_meta["SPLIT"], all_meta["SYNTHETIC"]])
    counts.to_csv(data_dir("split_label_counts.csv"))
    report["rows_by_split_source"] = {f"{a}|{b}": int(v) for (a, b), v in meta.groupby(["SPLIT", "SOURCE"]).size().items()}
    report["n_output_classes"] = len(labels)
    with open(data_dir("data_report.json"), "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    print(f"Data ready: {len(labels)} classes, {len(features)} genes")


def load_samples():
    return pd.read_csv(data_dir("samples.csv"))


def load_labels():
    with open(data_dir("labels.json")) as fh:
        return json.load(fh)


# ---------- Pair Creation Logic ------------
def create_pairs_by_logic(labels, is_metastatic, seed, max_pairs, overrepresent_metastatic=False):
    labels = np.asarray(labels)
    positive, negative = [], []
    use_pos, use_neg = defaultdict(int), defaultdict(int)
    by_label = {l: np.where(labels == l)[0].tolist() for l in sorted(set(labels))}
    limit = max_usage_per_sample_posneg

    for label, idx in by_label.items():
        idx = list(idx)
        np.random.RandomState(seed).shuffle(idx)
        for i in range(len(idx)):
            for j in range(i + 1, len(idx)):
                if use_pos[idx[i]] >= limit:
                    break
                if use_pos[idx[j]] < limit:
                    positive.append((idx[i], idx[j]))
                    use_pos[idx[i]] += 1
                    use_pos[idx[j]] += 1
            if len(positive) >= max_pairs // 2:
                break
        if len(positive) >= max_pairs // 2:
            break

    for l1, l2 in combinations(by_label, 2):
        idx1, idx2 = list(by_label[l1]), list(by_label[l2])
        rng = np.random.RandomState(seed)
        rng.shuffle(idx1)
        rng.shuffle(idx2)
        for a in idx1:
            if use_neg[a] >= limit:
                continue
            for b in idx2:
                if use_neg[a] >= limit:
                    break
                if use_neg[b] < limit:
                    negative.append((a, b))
                    use_neg[a] += 1
                    use_neg[b] += 1
            if len(negative) >= max_pairs // 2:
                break
        if len(negative) >= max_pairs // 2:
            break

    if overrepresent_metastatic:
        met = np.where(is_metastatic)[0]
        non_met = np.where(~np.asarray(is_metastatic))[0]
        count = defaultdict(int)
        have_pos, have_neg = set(positive), set(negative)
        for lab in sorted(set(labels[met])):
            m = met[labels[met] == lab].tolist()
            for i in range(len(m)):
                for j in range(i + 1, len(m)):
                    if count[m[i]] < metastatic_pos_neg_count and count[m[j]] < metastatic_pos_neg_count:
                        if (m[i], m[j]) not in have_pos:
                            positive.append((m[i], m[j]))
                        count[m[i]] += 1
                        count[m[j]] += 1
        rng = np.random.RandomState(seed)
        for m in met:
            for n in rng.choice(non_met, size=metastatic_pos_neg_count, replace=False):
                if (int(m), int(n)) not in have_neg:
                    negative.append((int(m), int(n)))

    pairs = positive + negative
    y = [1] * len(positive) + [0] * len(negative)
    pairs, y = shuffle(pairs, y, random_state=seed)
    return np.array(pairs[:max_pairs]), np.array(y[:max_pairs])


# ---------- PairGenerator and Model Builders ------------
if PHASE in ("embed", "meta", "export"):
    class PairGenerator(Sequence):
        def __init__(self, pairs, labels, X, batch_size=64, augment=False, indices=None):
            self.pairs = pairs if indices is None else pairs[indices]
            self.labels = labels if indices is None else labels[indices]
            self.X, self.batch_size, self.augment = X, batch_size, augment

        def __len__(self):
            return int(np.ceil(len(self.pairs) / self.batch_size))

        def _batch(self, i):
            s = slice(i * self.batch_size, (i + 1) * self.batch_size)
            p = self.pairs[s]
            a, b = self.X[p[:, 0]].copy(), self.X[p[:, 1]].copy()
            if self.augment:
                a += np.random.normal(0, 0.01, a.shape).astype(np.float32)
                b += np.random.normal(0, 0.01, b.shape).astype(np.float32)
            return a, b, self.labels[s]

        def __getitem__(self, i):
            a, b, y = self._batch(i)
            return (a, b), y


    class PairGeneratorCAE(PairGenerator):
        def __getitem__(self, i):
            a, b, y = self._batch(i)
            return (a, b), (y.reshape(-1, 1).astype("float32"), a, b)


    def create_siamese_network(input_dim, seed):
        init = tf.keras.initializers.HeNormal(seed=seed)
        inp = layers.Input(shape=(input_dim,))
        x = layers.Dense(256, activation="relu", kernel_initializer=init, kernel_regularizer=regularizers.l2(1e-5))(inp)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(128, activation="relu", kernel_initializer=init, kernel_regularizer=regularizers.l2(1e-5))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(64, activation="relu", kernel_initializer=init, kernel_regularizer=regularizers.l2(1e-5))(x)
        x = layers.BatchNormalization()(x)
        base = models.Model(inp, x, name="snn_base")
        a, b = layers.Input(shape=(input_dim,)), layers.Input(shape=(input_dim,))
        dist = layers.Lambda(lambda t: K.abs(t[0] - t[1]))([base(a), base(b)])
        out_ = layers.Dense(1, activation="sigmoid")(dist)
        model = models.Model([a, b], out_)
        model.compile(loss="binary_crossentropy", optimizer=optimizers.Adam(learning_rate=1e-4), metrics=["accuracy"])
        return model, base


    def contrastive_loss(y_true, dist):
        y_true = K.cast(y_true, dist.dtype)
        return K.mean(y_true * K.square(dist) + (1.0 - y_true) * K.square(K.maximum(1.0 - dist, 0)))


    def create_dual_loss_cae(input_dim, seed):
        init = tf.keras.initializers.HeNormal(seed=seed)
        reg = regularizers.l2(1e-4)
        inp = layers.Input(shape=(input_dim,))
        x = layers.Dense(256, activation="relu", kernel_initializer=init, kernel_regularizer=reg)(inp)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(128, activation="relu", kernel_initializer=init, kernel_regularizer=reg)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(64, activation="tanh", kernel_initializer=init, kernel_regularizer=reg)(x)
        enc_out = layers.Lambda(lambda t: K.l2_normalize(t, axis=1))(x)
        encoder = models.Model(inp, enc_out, name="encoder")

        dec_in = layers.Input(shape=(64,))
        d = layers.Dense(128, activation="relu", kernel_initializer=init, kernel_regularizer=reg)(dec_in)
        d = layers.BatchNormalization()(d)
        d = layers.Dropout(0.5)(d)
        d = layers.Dense(256, activation="relu", kernel_initializer=init, kernel_regularizer=reg)(d)
        d = layers.BatchNormalization()(d)
        dec_out = layers.Dense(input_dim, activation="linear", kernel_initializer=init)(d)
        decoder = models.Model(dec_in, dec_out, name="decoder")

        a, b = layers.Input(shape=(input_dim,)), layers.Input(shape=(input_dim,))
        ea, eb = encoder(a), encoder(b)
        dist = layers.Lambda(lambda t: K.sqrt(K.maximum(K.sum(K.square(t[0] - t[1]), axis=1, keepdims=True), K.epsilon())))([ea, eb])
        model = models.Model([a, b], [dist, decoder(ea), decoder(eb)], name="cae_dual")
        model.compile(loss=[contrastive_loss, "mse", "mse"], loss_weights=[1.0, 1.0, 1.0], optimizer=optimizers.Adam(learning_rate=1e-4))
        return model, encoder, decoder


    def create_meta_learner(h1, h2, h3, dropout_rate, l2_reg, optimizer, input_dim, num_classes):
        inp = layers.Input(shape=(input_dim,))
        x = inp
        for h in (h1, h2, h3):
            x = layers.Dense(h, activation="relu", kernel_regularizer=regularizers.l2(l2_reg))(x)
            x = layers.Dropout(dropout_rate)(x)
        out_ = layers.Dense(num_classes, activation="softmax")(x)
        m = models.Model(inp, out_)
        m.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
        return m


    def get_hard_examples_snn(model, gen, mode, top=0.15, min_n=64 * 50):
        hard = []
        for i in range(len(gen)):
            a, b, y = gen._batch(i)
            p = np.clip(model.predict_on_batch([a, b]).ravel().astype(np.float64), 1e-7, 1 - 1e-7)
            bce = -(y * np.log(p) + (1 - y) * np.log(1 - p))
            measure = 1.0 - np.abs(p - 0.5) if mode == "uncertainty" else np.abs(p - 0.5)
            hard.append(bce + measure)
        return top_hard(np.concatenate(hard), top, min_n)


    def get_hard_examples_cae(model, gen, mode, top=0.15, min_n=64 * 50):
        hard = []
        for i in range(len(gen)):
            a, b, y = gen._batch(i)
            dist, da, db = model.predict_on_batch([a, b])
            dist = dist.ravel().astype(np.float64)
            c = np.where(y == 1, dist ** 2, np.maximum(1.0 - dist, 0) ** 2)
            sample = c + np.mean((a - da) ** 2, axis=1) + np.mean((b - db) ** 2, axis=1)
            measure = 1.0 - np.abs(dist - 0.5) if mode == "uncertainty" else np.abs(dist - 0.5)
            hard.append(sample + measure)
        return top_hard(np.concatenate(hard), top, min_n)


    def top_hard(values, top, min_n):
        n = min(max(int(top * len(values)), min_n), len(values))
        return np.argsort(values)[::-1][:n]


def plot_train_val_loss(history, starts, title, path):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(history["loss"], label="Train")
    ax.plot(history["val_loss"], label="Validation")
    for s, c, t in zip(starts, ("red", "gold", "black"), ("Uncertainty HEM", "Overconfidence HEM", "Final phase")):
        ax.axvline(s - 0.5, color=c, linestyle=":", label=t)
    ax.set(xlabel="Epoch", ylabel="Loss", title=title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ---------- SNN & CAE Training ------------
def run_phases(model, gen_train, gen_val, pairs, y, X, epochs, hard_fn, gen_cls):
    init, unc, over, final = epochs
    hist = model.fit(gen_train, validation_data=gen_val, epochs=init, verbose=2).history
    for mode, n in (("uncertainty", unc), ("overconfident", over)):
        for _ in range(n):
            idx = hard_fn(model, gen_train, mode)
            hg = gen_cls(pairs, y, X, batch_size=64, augment=True, indices=idx)
            h = model.fit(hg, validation_data=gen_val, epochs=1, verbose=2).history
            for k, v in h.items():
                hist[k].extend(v)
    h = model.fit(gen_train, validation_data=gen_val, epochs=final, verbose=2).history
    for k, v in h.items():
        hist[k].extend(v)
    return {k: [float(x) for x in v] for k, v in hist.items()}


def train_embedders(seed, meta, X):
    if os.path.exists(seed_dir(seed, "metrics.json")):
        print(f"Seed {seed}: SNN and CAE already trained.")
        return
    tf.keras.utils.set_random_seed(seed)
    start = time.time()
    tr = meta["SPLIT"].values == "train"
    va = meta["SPLIT"].values == "val"
    te = meta["SPLIT"].values == "test"
    Xtr, Xva, Xte = X[tr], X[va], X[te]
    ytr, yva, yte = meta["LABEL"].values[tr], meta["LABEL"].values[va], meta["LABEL"].values[te]
    mtr = meta["SOURCE"].values[tr] == "METASTATIC"

    pair_file = seed_dir(seed, "pairs.npz")
    if os.path.exists(pair_file):
        z = np.load(pair_file)
        ptr, ltr, pva, lva, pte, lte = (z[k] for k in ("ptr", "ltr", "pva", "lva", "pte", "lte"))
    else:
        ptr, ltr = create_pairs_by_logic(ytr, mtr, seed, max_pairs, overrepresent_metastatic=True)
        pva, lva = create_pairs_by_logic(yva, np.zeros(len(yva), bool), seed, max_pairs)
        pte, lte = create_pairs_by_logic(yte, np.zeros(len(yte), bool), seed, max_pairs)
        np.savez_compressed(pair_file, ptr=ptr, ltr=ltr, pva=pva, lva=lva, pte=pte, lte=lte)
    wrong = int((ytr[ptr[ltr == 1][:, 0]] != ytr[ptr[ltr == 1][:, 1]]).sum())
    if wrong:
        raise RuntimeError(f"{wrong} positive pairs join different labels")
    print(f"Seed {seed}: {len(ptr):,} train pairs ({int(ltr.sum()):,} positive), {len(pva):,} validation pairs", flush=True)

    d = Xtr.shape[1]
    metrics = {"seed": seed, "train_pairs": int(len(ptr)), "val_pairs": int(len(pva))}

    if os.path.exists(seed_dir(seed, "snn_full.weights.h5")):
        snn, base = create_siamese_network(d, seed)
        snn.load_weights(seed_dir(seed, "snn_full.weights.h5"))
    else:
        with timer(f"SNN training, seed {seed}"):
            snn, base = create_siamese_network(d, seed)
            g_tr = PairGenerator(ptr, ltr, Xtr, batch_size=64, augment=True)
            g_va = PairGenerator(pva, lva, Xva, batch_size=64)
            hist = run_phases(snn, g_tr, g_va, ptr, ltr, Xtr, SNN_EPOCHS, get_hard_examples_snn, PairGenerator)
        snn.save_weights(seed_dir(seed, "snn_full.weights.h5"))
        with open(seed_dir(seed, "snn_history.json"), "w") as fh:
            json.dump(hist, fh)
        e = SNN_EPOCHS
        plot_train_val_loss(hist, (e[0], e[0] + e[1], e[0] + e[1] + e[2]), f"SNN loss, seed {seed}", seed_dir(seed, "snn_loss.png"))
    base.save(seed_dir(seed, "snn_base.h5"), include_optimizer=False)
    pred = snn.predict([Xte[pte[:, 0]], Xte[pte[:, 1]]], batch_size=4096, verbose=0).ravel()
    metrics["snn_test_pair_accuracy"] = float(accuracy_score(lte, pred >= 0.5))
    metrics["snn_test_pair_auc"] = float(roc_auc_score(lte, pred))

    tf.keras.utils.set_random_seed(seed + 1)
    if os.path.exists(seed_dir(seed, "cae_full.weights.h5")):
        cae, enc, dec = create_dual_loss_cae(d, seed)
        cae.load_weights(seed_dir(seed, "cae_full.weights.h5"))
    else:
        with timer(f"CAE training, seed {seed}"):
            cae, enc, dec = create_dual_loss_cae(d, seed)
            g_tr = PairGeneratorCAE(ptr, ltr, Xtr, batch_size=64, augment=True)
            g_va = PairGeneratorCAE(pva, lva, Xva, batch_size=64)
            hist = run_phases(cae, g_tr, g_va, ptr, ltr, Xtr, CAE_EPOCHS, get_hard_examples_cae, PairGeneratorCAE)
        cae.save_weights(seed_dir(seed, "cae_full.weights.h5"))
        with open(seed_dir(seed, "cae_history.json"), "w") as fh:
            json.dump(hist, fh)
        e = CAE_EPOCHS
        plot_train_val_loss(hist, (e[0], e[0] + e[1], e[0] + e[1] + e[2]), f"CAE loss, seed {seed}", seed_dir(seed, "cae_loss.png"))
    enc.save_weights(seed_dir(seed, "cae_encoder.weights.h5"))
    dec.save_weights(seed_dir(seed, "cae_decoder.weights.h5"))
    rec = dec.predict(enc.predict(Xte, batch_size=4096, verbose=0), batch_size=4096, verbose=0)
    metrics["cae_test_r2_all_values"] = float(r2_score(Xte.ravel(), rec.ravel()))

    emb = np.hstack([base.predict(X, batch_size=4096, verbose=0), enc.predict(X, batch_size=4096, verbose=0)])
    np.save(seed_dir(seed, "embeddings_all.npy"), emb.astype(np.float32))
    metrics["minutes_total"] = round((time.time() - start) / 60, 1)
    with open(seed_dir(seed, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"Seed {seed} embedders complete:", metrics, flush=True)
    tf.keras.backend.clear_session()


def phase_embed():
    meta = load_samples()
    X = np.load(data_dir("X_scaled_features.npy"))
    for seed in SEED_LIST:
        train_embedders(seed, meta, X)


# ---------- Base Classifiers ------------
def make_clf(name, params, seed, n_classes, n_jobs=1):
    if name == "LogisticRegression":
        return LogisticRegression(**params, random_state=seed)
    if name == "KNN":
        return KNeighborsClassifier(**params, n_jobs=n_jobs)
    if name == "RandomForest":
        return RandomForestClassifier(**params, random_state=seed, n_jobs=n_jobs)
    if name == "SVM":
        return SVC(**params, probability=True, random_state=seed)
    return XGBClassifier(**params, eval_metric="mlogloss", verbosity=0, objective="multi:softprob", num_class=n_classes,
                         tree_method="hist", random_state=seed, n_jobs=n_jobs, use_label_encoder=False)


def align(proba, seen, n):
    res = np.zeros((proba.shape[0], n), dtype=np.float32)
    res[:, seen] = proba
    return res


def fold_task(j, name, params, X, y, tr_i, va_i, seed, n):
    ytr = y[tr_i]
    seen = np.unique(ytr)
    clf = make_clf(name, params, seed, len(seen))
    clf.fit(X[tr_i], np.searchsorted(seen, ytr))
    return j, va_i, align(clf.predict_proba(X[va_i]), seen, n)


def fit_task(name, params, X, y, seed):
    seen = np.unique(y)
    clf = make_clf(name, params, seed, len(seen))
    clf.fit(X, np.searchsorted(seen, y))
    clf.seen_classes_ = seen
    return name, clf


def tune(X, y, seed):
    if os.path.exists(hyperparameters_txt_file):
        with open(hyperparameters_txt_file) as fh:
            return json.load(fh)
    best = {}
    n = len(np.unique(y))
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    for name, grid in GRIDS.items():
        partial = out("03_stacking", f"tuned_{name}.json")
        if os.path.exists(partial):
            with open(partial) as fh:
                best[name] = json.load(fh)
            continue
        with timer(f"GridSearchCV {name}"):
            est = make_clf(name, {}, seed, n, n_jobs=1)
            if name == "SVM":
                est.set_params(probability=False)
            gs = GridSearchCV(est, grid, cv=cv, scoring="accuracy", n_jobs=-1, error_score=np.nan, verbose=1)
            gs.fit(X, y)
        best[name] = gs.best_params_
        pd.DataFrame(gs.cv_results_).to_csv(out("03_stacking", f"grid_results_{name}.csv"), index=False)
        with open(partial, "w") as fh:
            json.dump(gs.best_params_, fh, indent=1)
        print(f"  {name}: {gs.best_params_} (CV accuracy {gs.best_score_:.4f})", flush=True)
    with open(hyperparameters_txt_file, "w") as fh:
        json.dump(best, fh, indent=1)
    return best


def split_design(meta, y, seed):
    tr_real = np.where((meta["SPLIT"] == "train") & ~meta["SYNTHETIC"])[0]
    va = np.where(meta["SPLIT"] == "val")[0]
    counts = pd.Series(y[va]).value_counts()
    can = np.isin(y[va], counts[counts >= 2].index)
    a, b = train_test_split(va[can], test_size=0.3, random_state=seed, stratify=y[va][can])
    rest = np.concatenate([a, va[~can]])
    return np.concatenate([tr_real, rest]), b


def phase_bases(variant, seed):
    feats_file = variant_dir(variant, seed, "meta_features.npz")
    if os.path.exists(feats_file):
        return
    meta = load_samples()
    labels = load_labels()
    emb = np.load(seed_dir(seed, "embeddings_all.npy"))
    if variant == "snn_only":
        emb = emb[:, :64]
    elif variant == "cae_only":
        emb = emb[:, 64:]
    y = np.searchsorted(labels, meta["LABEL"].values)
    n = len(labels)

    base_idx, val_idx = split_design(meta, y, seed)
    test_idx = np.where(meta["SPLIT"] == "test")[0]
    if set(base_idx) & set(test_idx) or set(val_idx) & set(test_idx) or set(base_idx) & set(val_idx):
        raise RuntimeError("Stacking sets overlap")
    design = {"base": len(base_idx), "meta_val": len(val_idx), "test": len(test_idx),
              "base_classes": int(len(np.unique(y[base_idx]))), "synthetic_in_base": int(meta["SYNTHETIC"].values[base_idx].sum())}
    print(f"{variant}, seed {seed}:", design, flush=True)

    params = tune(emb[base_idx], y[base_idx], seed)
    Xb, yb = emb[base_idx], y[base_idx]

    with timer(f"Base classifiers, {variant}, seed {seed}"):
        folds = list(StratifiedKFold(5, shuffle=True, random_state=seed).split(Xb, yb))
        results = Parallel(n_jobs=-2, backend="loky")(
            delayed(fold_task)(j, name, params[name], Xb, yb, tr, va, seed, n)
            for j, name in enumerate(CLASSIFIERS) for tr, va in folds)
        F_tr = np.zeros((len(yb), n * len(CLASSIFIERS)), dtype=np.float32)
        for j, va, proba in results:
            F_tr[va, j * n:(j + 1) * n] = proba

        fitted = dict(Parallel(n_jobs=5, backend="loky")(delayed(fit_task)(name, params[name], Xb, yb, seed) for name in CLASSIFIERS))
    clfs = {name: fitted[name] for name in CLASSIFIERS}
    with open(variant_dir(variant, seed, "base_classifiers.pkl"), "wb") as fh:
        pickle.dump(clfs, fh)
    feats = lambda X: np.hstack([align(c.predict_proba(X), c.seen_classes_, n) for c in clfs.values()])
    np.savez_compressed(feats_file, F_tr=F_tr, y_tr=yb, F_va=feats(emb[val_idx]), y_va=y[val_idx], F_te=feats(emb[test_idx]), y_te=y[test_idx])
    with open(variant_dir(variant, seed, "design.json"), "w") as fh:
        json.dump(design, fh, indent=1)


# ---------- Meta-Learner ------------
def phase_meta(variant, seed):
    if os.path.exists(variant_dir(variant, seed, "test_proba.npy")):
        return
    z = np.load(variant_dir(variant, seed, "meta_features.npz"))
    F_tr, yb, F_va, yv, F_te = z["F_tr"], z["y_tr"], z["F_va"], z["y_va"], z["F_te"]
    n = len(load_labels())
    rows, best, best_acc = [], None, -1.0
    for i, (h1, h2, h3, dr, l2, opt, bs, ep) in enumerate(restricted_archs, 1):
        tf.keras.utils.set_random_seed(seed + i)
        m = create_meta_learner(h1, h2, h3, dr, l2, opt, F_tr.shape[1], n)
        m.fit(F_tr, yb, validation_data=(F_va, yv), epochs=ep, batch_size=bs, verbose=0)
        acc = float((m.predict(F_va, batch_size=4096, verbose=0).argmax(1) == yv).mean())
        rows.append({"arch": i, "h1": h1, "h2": h2, "h3": h3, "dropout": dr, "l2": l2, "batch": bs, "epochs": ep, "val_accuracy": acc})
        m.save(variant_dir(variant, seed, f"meta_learner_{i}.h5"), include_optimizer=False)
        if acc > best_acc:
            best, best_acc = i, acc
    pd.DataFrame(rows).to_csv(variant_dir(variant, seed, "meta_selection_on_validation.csv"), index=False)
    with open(variant_dir(variant, seed, "best_arch.txt"), "w") as fh:
        fh.write(str(best))
    chosen = tf.keras.models.load_model(variant_dir(variant, seed, f"meta_learner_{best}.h5"), compile=False)
    np.save(variant_dir(variant, seed, "test_proba.npy"), chosen.predict(F_te, batch_size=4096, verbose=0).astype(np.float32))


# ---------- Test Evaluation ------------
def restrict_to_test_classes(proba, test_classes):
    p = proba.copy()
    p[:, ~np.isin(np.arange(p.shape[1]), test_classes)] = 0
    return p / np.maximum(p.sum(axis=1, keepdims=True), 1e-12)


def wilson(k, n, z=1.96):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def evaluate(y, proba, labels, sources):
    pred = proba.argmax(1)
    res = {}

    def block(mask, tag):
        yt, p, pb = y[mask], pred[mask], proba[mask]
        order = np.argsort(-pb, axis=1)
        present = np.unique(yt)
        res[f"{tag}_n"] = int(mask.sum())
        res[f"{tag}_accuracy"] = accuracy_score(yt, p)
        res[f"{tag}_f1_weighted"] = f1_score(yt, p, average="weighted", zero_division=0)
        res[f"{tag}_f1_macro"] = f1_score(yt, p, labels=present, average="macro", zero_division=0)
        res[f"{tag}_recall_weighted"] = recall_score(yt, p, average="weighted", zero_division=0)
        res[f"{tag}_recall_macro"] = recall_score(yt, p, labels=present, average="macro", zero_division=0)
        res[f"{tag}_precision_weighted"] = precision_score(yt, p, average="weighted", zero_division=0)
        for k in (2, 3, 5):
            res[f"{tag}_top{k}"] = float(np.mean([t in o[:k] for t, o in zip(yt, order)]))
        aucs = [roc_auc_score(y == c, proba[:, c]) for c in present if 0 < (y == c).sum() < len(y)]
        res[f"{tag}_auc_macro_ovr"] = float(np.mean(aucs)) if aucs else float("nan")

    block(np.ones(len(y), bool), "all")
    for s, tag in (("GTEX", "gtex"), ("TCGA", "tcga"), ("METASTATIC", "metastatic")):
        block(sources == s, tag)
    res["balanced_mean_of_source_accuracies"] = float(np.mean([res["gtex_accuracy"], res["tcga_accuracy"], res["metastatic_accuracy"]]))

    names = np.array(labels)
    root = lambda a: np.array([x.split("_")[0] for x in a])
    m = sources == "METASTATIC"
    res["metastatic_cancer_type_accuracy"] = float(np.mean(root(names[y[m]]) == root(names[pred[m]])))

    per = []
    order = np.argsort(-proba, axis=1)
    for c in np.unique(y):
        mask = y == c
        k = mask.sum()
        acc = (pred[mask] == c).mean()
        lo, hi = wilson(acc * k, k)
        tp, fp = int(((pred == c) & mask).sum()), int(((pred == c) & ~mask).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        per.append({"label": names[c], "n": int(k), "accuracy": acc, "wilson_low": lo, "wilson_high": hi,
                    "top2": float((order[mask, :2] == c).any(1).mean()), "top3": float((order[mask, :3] == c).any(1).mean()),
                    "top5": float((order[mask, :5] == c).any(1).mean()), "precision": prec,
                    "f1": 2 * prec * acc / (prec + acc) if prec + acc else 0.0, "auc": roc_auc_score(y == c, proba[:, c])})
    return res, pd.DataFrame(per)


def phase_score(variant, seed):
    meta = load_samples()
    labels = load_labels()
    te = meta[meta["SPLIT"] == "test"]
    z = np.load(variant_dir(variant, seed, "meta_features.npz"))
    y, F_te = z["y_te"], z["F_te"]
    n = len(labels)
    test_classes = np.unique(y)
    proba = restrict_to_test_classes(np.load(variant_dir(variant, seed, "test_proba.npy")), test_classes)
    res, per = evaluate(y, proba, labels, te["SOURCE"].values)
    with open(variant_dir(variant, seed, "design.json")) as fh:
        res.update(json.load(fh))
    with open(variant_dir(variant, seed, "best_arch.txt")) as fh:
        res.update(seed=seed, variant=variant, best_arch=int(fh.read()))
    res["val_accuracy"] = float(pd.read_csv(variant_dir(variant, seed, "meta_selection_on_validation.csv"))["val_accuracy"].max())
    for j, name in enumerate(CLASSIFIERS):
        F = F_te[:, j * n:(j + 1) * n].copy()
        F[:, ~np.isin(np.arange(n), test_classes)] = -1
        res[f"base_{name}_accuracy"] = float(accuracy_score(y, F.argmax(1)))
    with open(variant_dir(variant, seed, "test_metrics.json"), "w") as fh:
        json.dump(res, fh, indent=2, default=float)
    per.to_csv(variant_dir(variant, seed, "test_metrics_per_class.csv"), index=False)
    print(f"{variant}, seed {seed}: accuracy {res['all_accuracy']:.4f} | GTEx {res['gtex_accuracy']:.4f} | "
          f"TCGA {res['tcga_accuracy']:.4f} | metastatic {res['metastatic_accuracy']:.4f}", flush=True)


# ---------- Export of the deployed model (seed 42) ------------
def numpy_forward():
    def dense(x, w, p, i):
        return x @ w[f"{p}_{i}_0"] + w[f"{p}_{i}_1"]

    def bn(x, w, p, i):
        gamma, beta, mean, var = (w[f"{p}_{i}_{j}"] for j in range(4))
        return (x - mean) / np.sqrt(var + 1e-3) * gamma + beta

    relu = lambda x: np.maximum(x, 0)

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

    return snn, cae, meta


def export_networks(seed):
    if os.path.exists(model_dir("nets_numpy.npz")):
        return
    meta_df = load_samples()
    X = np.load(data_dir("X_scaled_features.npy"))
    Xs = X[(meta_df["SPLIT"] == "test").values][:400]
    d = X.shape[1]
    n = len(load_labels())

    snn_model, snn_base = create_siamese_network(d, seed)
    snn_model.load_weights(seed_dir(seed, "snn_full.weights.h5"))
    snn_model.save(model_dir("snn_model.h5"), include_optimizer=False)
    cae_model, cae_encoder, cae_decoder = create_dual_loss_cae(d, seed)
    cae_model.load_weights(seed_dir(seed, "cae_full.weights.h5"))
    cae_model.save(model_dir("cae_autoencoder_dual_loss.h5"), include_optimizer=False)
    cae_encoder.save(model_dir("cae_encoder.h5"), include_optimizer=False)
    cae_decoder.save(model_dir("cae_decoder.h5"), include_optimizer=False)
    with open(variant_dir("primary", seed, "best_arch.txt")) as fh:
        best = fh.read().strip()
    meta_m = tf.keras.models.load_model(variant_dir("primary", seed, f"meta_learner_{best}.h5"), compile=False)
    meta_m.save(model_dir(f"best_meta_learner_{best}.h5"), include_optimizer=False)

    def flat(model, prefix):
        arrays, kinds = {}, []
        for i, l in enumerate([l for l in model.layers if l.get_weights()]):
            kinds.append(type(l).__name__)
            for j, w in enumerate(l.get_weights()):
                arrays[f"{prefix}_{i}_{j}"] = w
        arrays[f"{prefix}_kinds"] = np.array(kinds)
        return arrays

    np.savez(model_dir("nets_numpy.npz"), **flat(snn_base, "snn"), **flat(cae_encoder, "cae"), **flat(meta_m, "meta"))
    snn, cae, meta = numpy_forward()
    w = dict(np.load(model_dir("nets_numpy.npz")))
    F = np.random.default_rng(0).random((200, n * len(CLASSIFIERS)), dtype=np.float32)
    errs = {"snn": float(np.abs(snn(w, Xs) - snn_base(Xs, training=False).numpy()).max()),
            "cae": float(np.abs(cae(w, Xs) - cae_encoder(Xs, training=False).numpy()).max()),
            "meta": float(np.abs(meta(w, F) - meta_m(F, training=False).numpy()).max())}
    print("NumPy versus TensorFlow, largest difference:", errs, flush=True)
    if max(errs.values()) > 1e-3:
        os.remove(model_dir("nets_numpy.npz"))
        raise RuntimeError(f"NumPy forward pass differs from TensorFlow: {errs}")


def export_model_files(seed):
    meta = load_samples()
    labels = load_labels()
    features = pd.read_csv(data_dir("features.csv"))["gene"].tolist()
    common = pd.read_csv(data_dir("common_genes.csv"))["gene"].tolist()
    shutil.copy2(data_dir("scaler.pkl"), model_dir("BalancedMetastaticStandardScaler.pkl"))
    shutil.copy2(variant_dir("primary", seed, "base_classifiers.pkl"), model_dir("trained_base_classifiers.pkl"))
    shutil.copy2(hyperparameters_txt_file, model_dir("base_classifiers_hyperparameters.txt"))
    with open(model_dir("feature_sets.pkl"), "wb") as fh:
        pickle.dump({"FullLasso": features}, fh)
    with open(model_dir("label_encoder.pkl"), "wb") as fh:
        pickle.dump(LabelEncoder().fit(labels), fh)

    X = np.load(data_dir("X_log2tpm_all_genes.npy"), mmap_mode="r")
    real = meta.iloc[: X.shape[0]]
    fidx = [common.index(g) for g in features]
    for name, mask in (("Cancer", real.SOURCE.isin(["TCGA", "METASTATIC"])), ("Healthy", real.SOURCE == "GTEX")):
        rows = np.where((real.SPLIT == "train").values & mask.values)[0]
        sub = np.asarray(X[rows][:, fidx], dtype=np.float64)
        pd.Series(np.nanmedian(sub, axis=0), index=features).to_pickle(model_dir(f"Median_Gene_Values_{name}.pkl"))
        pd.Series(np.nanstd(sub, axis=0), index=features).to_pickle(model_dir(f"Std_Gene_Values_{name}.pkl"))

    real_rows = meta[~meta["SYNTHETIC"]]
    counts = pd.DataFrame({"TrainCount": real_rows[real_rows.SPLIT == "train"].LABEL.value_counts(),
                           "SMOTE_Generated": meta[meta["SYNTHETIC"]].LABEL.value_counts(),
                           "ValCount": real_rows[real_rows.SPLIT == "val"].LABEL.value_counts(),
                           "TestCount": real_rows[real_rows.SPLIT == "test"].LABEL.value_counts()}).reindex(labels).fillna(0).astype(int)
    counts.index.name = "Label"
    counts.to_csv(model_dir("split_label_counts.csv"))

    sel = pd.read_csv(variant_dir("primary", seed, "meta_selection_on_validation.csv"))
    sel.rename(columns={"arch": "Iteration", "dropout": "dropout_rate", "l2": "l2_reg", "batch": "batch_size", "val_accuracy": "Val_Accuracy"}).to_csv(
        model_dir("meta_learner_restricted_results.csv"), index=False)

    res = json.load(open(variant_dir("primary", seed, "test_metrics.json")))
    per = pd.read_csv(variant_dir("primary", seed, "test_metrics_per_class.csv"))
    rows = []
    for tag, name in (("all", "AllSources"), ("tcga", "TCGA"), ("gtex", "GTEx"), ("metastatic", "Metastatic")):
        lo, hi = wilson(res[f"{tag}_accuracy"] * res[f"{tag}_n"], res[f"{tag}_n"])
        rows.append({"Set": name, "N": res[f"{tag}_n"], "Accuracy": res[f"{tag}_accuracy"], "CI_low": lo, "CI_high": hi, "Top-2": res[f"{tag}_top2"],
                     "Top-3": res[f"{tag}_top3"], "Top-5": res[f"{tag}_top5"], "F1": res[f"{tag}_f1_weighted"], "Precision": res[f"{tag}_precision_weighted"],
                     "Recall": res[f"{tag}_recall_weighted"], "ROC-AUC": res[f"{tag}_auc_macro_ovr"]})
    for _, r in per.iterrows():
        rows.append({"Set": r["label"], "N": r["n"], "Accuracy": r["accuracy"], "CI_low": r["wilson_low"], "CI_high": r["wilson_high"], "Top-2": r["top2"],
                     "Top-3": r["top3"], "Top-5": r["top5"], "F1": r["f1"], "Precision": r["precision"], "Recall": r["accuracy"], "ROC-AUC": r["auc"]})
    pd.DataFrame(rows).round(4).to_csv(model_dir("test_split_metrics.csv"), index=False)


# ---------- Aggregate Metrics Across Seeds ------------
def aggregate_metrics():
    all_metrics = []
    for seed in SEED_LIST:
        f = variant_dir("primary", seed, "test_metrics.json")
        if os.path.exists(f):
            all_metrics.append(json.load(open(f)))
    if not all_metrics:
        print("No seed metrics found.")
        return
    df = pd.DataFrame(all_metrics)
    line = lambda key, name: (f"{name:20s}: {df[key].mean():.4f} ± {df[key].std(ddof=0):.4f}  "
                              f"(min={df[key].min():.4f}, max={df[key].max():.4f})\n")
    with open(model_dir("aggregated_metrics_all_seeds.txt"), "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("AGGREGATED METRICS ACROSS ALL SEEDS\n")
        f.write("=" * 80 + "\n")
        f.write(f"Seeds: {list(df['seed'])}\n")
        f.write(f"Number of seeds: {len(df)}\n")
        f.write(f"Classes scored: {len(pd.read_csv(variant_dir('primary', SEED_LIST[0], 'test_metrics_per_class.csv')))} labels with test samples\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        f.write("OVERALL METRICS (All Test Data)\n")
        f.write("-" * 80 + "\n")
        for key, name in (("all_accuracy", "Accuracy"), ("all_precision_weighted", "Precision"), ("all_recall_weighted", "Recall"),
                          ("all_f1_weighted", "F1"), ("all_top2", "Top2"), ("all_top3", "Top3"), ("all_top5", "Top5"), ("all_auc_macro_ovr", "Roc Auc")):
            f.write(line(key, name))
        f.write("\nSOURCE-SPECIFIC METRICS\n")
        f.write("=" * 80 + "\n\n")
        for tag in ("gtex", "tcga", "metastatic"):
            f.write(f"{tag.upper()} DATA\n")
            f.write("-" * 80 + "\n")
            for key, name in ((f"{tag}_accuracy", "Accuracy"), (f"{tag}_f1_weighted", "F1"), (f"{tag}_top2", "Top2")):
                f.write(line(key, name))
            f.write("\n")
        f.write("=" * 80 + "\n")
        f.write("PER-SEED BREAKDOWN\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"{'Seed':>6s}  {'Overall Acc':>12s}  {'Overall F1':>12s}  {'GTEx Acc':>10s}  {'TCGA Acc':>10s}  {'Met Acc':>10s}\n")
        f.write("-" * 80 + "\n")
        for _, m in df.iterrows():
            f.write(f"{int(m['seed']):>6d}  {m['all_accuracy']:>12.4f}  {m['all_f1_weighted']:>12.4f}  {m['gtex_accuracy']:>10.4f}  "
                    f"{m['tcga_accuracy']:>10.4f}  {m['metastatic_accuracy']:>10.4f}\n")
        f.write("\n" + "=" * 80 + "\n")
    print(open(model_dir("aggregated_metrics_all_seeds.txt"), encoding="utf-8").read())


def plot_embedding_comparison(seed):
    vals = {}
    for variant, name in (("snn_only", "SNN"), ("cae_only", "CAE"), ("primary", "Combined")):
        f = variant_dir(variant, seed, "test_metrics.json")
        if not os.path.exists(f):
            return
        r = json.load(open(f))
        vals[name] = (r["all_accuracy"], r["all_f1_weighted"])
    x = np.arange(len(vals))
    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(x - 0.175, [v[0] for v in vals.values()], width=0.35, label="Accuracy", color="blue")
    bars += ax.bar(x + 0.175, [v[1] for v in vals.values()], width=0.35, label="F1 Score", color="red")
    ax.set_ylim([0.75, 1.0])
    ax.set_xticks(x)
    ax.set_xticklabels(list(vals))
    ax.set(ylabel="Metric Value", xlabel="Embeddings Used", title="Informativeness of Generated Embeddings")
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    for rect in bars:
        ax.annotate(f"{rect.get_height():.3f}", xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out("03_stacking", "meta_embedding_comparison_barplot.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------- Unit Tests ------------
class SplitDataTests(unittest.TestCase):
    def setUp(self):
        self.meta = load_samples() if os.path.exists(data_dir("samples.csv")) else None

    def test_data_exists(self):
        self.assertIsNotNone(self.meta, "Prepared data not found")

    def test_no_overlap_in_splits(self):
        real = self.meta[~self.meta["SYNTHETIC"]]
        for col in ("SampleID", "GROUP_ID"):
            sets = {s: set(real.loc[real.SPLIT == s, col]) for s in ("train", "val", "test")}
            self.assertTrue(sets["train"].isdisjoint(sets["val"]), f"{col} overlap between train and val")
            self.assertTrue(sets["train"].isdisjoint(sets["test"]), f"{col} overlap between train and test")
            self.assertTrue(sets["val"].isdisjoint(sets["test"]), f"{col} overlap between val and test")

    def test_synthetic_only_in_train(self):
        self.assertTrue((self.meta.loc[self.meta["SYNTHETIC"], "SPLIT"] == "train").all())

    def test_test_labels_seen_in_train(self):
        self.assertTrue(set(self.meta.loc[self.meta.SPLIT == "test", "LABEL"]) <= set(self.meta.loc[self.meta.SPLIT == "train", "LABEL"]))


# ---------------- MAIN SCRIPT START ----------------
def run_step(*args):
    for attempt in range(1, 4):
        r = subprocess.run([sys.executable, os.path.abspath(__file__), "--phase", *[str(a) for a in args]])
        if r.returncode == 0:
            return
        print(f"Step {args} failed (exit {r.returncode}), attempt {attempt} of 3. Rerunning from its checkpoints...", flush=True)
    sys.exit(1)


def main():
    print(f"Project directory: {project_dir}")
    print(f"Output directory: {output_dir}")
    with timer("Full Pipeline Execution"):
        run_step("data")
        run_step("embed")
        for variant, seeds in (("primary", SEED_LIST), ("snn_only", SEED_LIST[:1]), ("cae_only", SEED_LIST[:1])):
            for seed in seeds:
                if os.path.exists(variant_dir(variant, seed, "test_metrics.json")):
                    continue
                for phase in ("bases", "meta", "score"):
                    run_step(phase, variant, seed)
        run_step("export", SEED_LIST[0])
        export_model_files(SEED_LIST[0])
        aggregate_metrics()
        plot_embedding_comparison(SEED_LIST[0])
    print("Running unit tests on splits:")
    unittest.main(argv=[''], exit=False)
    print("Script execution completed. The deployed model files are in", model_dir())


if __name__ == "__main__":
    if PHASE is None:
        main()
    elif PHASE == "data":
        prepare_data()
    elif PHASE == "embed":
        phase_embed()
    elif PHASE == "export":
        export_networks(int(sys.argv[3]))
    else:
        {"bases": phase_bases, "meta": phase_meta, "score": phase_score}[PHASE](sys.argv[3], int(sys.argv[4]))
