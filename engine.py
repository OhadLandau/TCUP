"""TCUP prediction engine: TPM in, class probabilities and the most influential genes out."""
import io
import math
import os
import pickle

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
TOP_N = 20


def _bh(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.minimum(q, 1)
    return out


def _dense(x, w, p, i):
    return x @ w[f"{p}_{i}_0"] + w[f"{p}_{i}_1"]


def _bn(x, w, p, i):
    gamma, beta, mean, var = (w[f"{p}_{i}_{j}"] for j in range(4))
    return (x - mean) / np.sqrt(var + 1e-3) * gamma + beta


def _relu(x):
    return np.maximum(x, 0)


def _snn(w, x):
    for i in (0, 2, 4):
        x = _bn(_relu(_dense(x, w, "snn", i)), w, "snn", i + 1)
    return x


def _cae(w, x):
    x = _bn(_relu(_dense(x, w, "cae", 0)), w, "cae", 1)
    x = _bn(_relu(_dense(x, w, "cae", 2)), w, "cae", 3)
    x = np.tanh(_dense(x, w, "cae", 4))
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def _meta(w, x):
    for i in range(3):
        x = _relu(_dense(x, w, "meta", i))
    logits = _dense(x, w, "meta", 3)
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


TCGA_NAMES = {
    "BLCA": "Bladder urothelial carcinoma", "BRCA": "Breast invasive carcinoma", "CESC": "Cervical squamous cell carcinoma",
    "COAD": "Colon adenocarcinoma", "HNSC": "Head and neck squamous cell carcinoma", "KIRC": "Kidney renal clear cell carcinoma",
    "LGG": "Lower grade glioma", "LIHC": "Liver hepatocellular carcinoma", "LUAD": "Lung adenocarcinoma",
    "PAAD": "Pancreatic adenocarcinoma", "PRAD": "Prostate adenocarcinoma", "SKCM": "Skin cutaneous melanoma",
    "STAD": "Stomach adenocarcinoma", "THCA": "Thyroid carcinoma", "UCEC": "Uterine corpus endometrial carcinoma",
}
GTEX_NAMES = {"Minor": "Minor salivary gland", "Small": "Small intestine", "Whole": "Whole blood"}
ALPHA = 0.05


def describe(label):
    code, source = label.rsplit("_", 1)
    code = code.replace(".csv", "")
    if source == "GTEx":
        return {"code": code, "source": "Healthy", "name": f"{GTEX_NAMES.get(code, code)} (normal tissue)", "tissue": GTEX_NAMES.get(code, code)}
    kind = "Metastatic" if source == "METASTATIC" else "Primary tumor"
    return {"code": code, "source": kind, "name": f"{TCGA_NAMES.get(code, code)}{' (metastasis)' if kind == 'Metastatic' else ''}", "tissue": TCGA_NAMES.get(code, code)}


def _gene_set(mc, suffix, samples):
    d, q = mc[f"AccuracyDrop{suffix}"].values, mc[f"FDR{suffix}"].values
    order = np.argsort(-d, kind="stable")
    genes = mc["Gene"].values
    return {"samples": int(samples), "significant": [genes[i] for i in order if q[i] < ALPHA],
            "top": [{"gene": genes[i], "drop": round(float(d[i]), 5), "fdr": float(q[i])} for i in order[:TOP_N]]}


class Engine:
    def __init__(self):
        with open(os.path.join(ROOT, "feature_sets.pkl"), "rb") as fh:
            self.features = pickle.load(fh)["FullLasso"]
        with open(os.path.join(ROOT, "label_encoder.pkl"), "rb") as fh:
            self.model_labels = list(pickle.load(fh).classes_)
        metrics = pd.read_csv(os.path.join(ROOT, "test_split_metrics.csv")).set_index("Set")
        counts = pd.read_csv(os.path.join(ROOT, "split_label_counts.csv")).set_index("Label")
        self.classes = []
        for lab in self.model_labels:
            if lab in metrics.index:
                r = metrics.loc[lab]
                self.classes.append({**describe(lab), "label": lab, "test_n": int(r["N"]), "test_accuracy": float(r["Accuracy"]),
                                     "ci_low": float(r["CI_low"]), "ci_high": float(r["CI_high"])})
        self.labels = [c["label"] for c in self.classes]
        self.keep = np.array([self.model_labels.index(l) for l in self.labels])
        self.class_info = {c["label"]: c for c in self.classes}
        test_metrics = {}
        for key, name in (("all", "AllSources"), ("tcga", "TCGA"), ("gtex", "GTEx"), ("metastatic", "Metastatic")):
            test_metrics.update({f"{key}_n": int(metrics.loc[name, "N"]), f"{key}_accuracy": float(metrics.loc[name, "Accuracy"]),
                                 f"{key}_top3": float(metrics.loc[name, "Top-3"])})
        healthy = counts.index.str.endswith("_GTEx")
        self.config = {"features": self.features, "classes": self.classes, "test_metrics": test_metrics, "model": "TCUP v2",
                       "reference_samples": {"cancer": int(counts.loc[~healthy, "TrainCount"].sum()), "healthy": int(counts.loc[healthy, "TrainCount"].sum())}}
        mc = pd.read_csv(os.path.join(ROOT, "monte_carlo_gene_importance_averaged.csv"))
        mc = mc.set_index("Gene").loc[self.features].reset_index()
        self.genes = {"overall": _gene_set(mc, "", test_metrics["all_n"]), "metastatic": _gene_set(mc, "_Metastatic", test_metrics["metastatic_n"]),
                      "by_class": {c["label"]: _gene_set(mc, f"_{c['label']}", c["test_n"]) for c in self.classes if c["test_n"] >= 5}}
        z = np.load(os.path.join(ROOT, "nets_numpy.npz"), allow_pickle=False)
        self.nets = {k: z[k] for k in z.files}
        self.ref = {}
        for key in ("cancer", "healthy"):
            self.ref[f"med_{key}"] = pd.read_pickle(os.path.join(ROOT, f"Median_Gene_Values_{key.capitalize()}.pkl")).loc[self.features].values.astype(np.float64)
            self.ref[f"sd_{key}"] = pd.read_pickle(os.path.join(ROOT, f"Std_Gene_Values_{key.capitalize()}.pkl")).loc[self.features].values.astype(np.float64)
        with open(os.path.join(ROOT, "BalancedMetastaticStandardScaler.pkl"), "rb") as fh:
            self.scaler = pickle.load(fh)
        with open(os.path.join(ROOT, "trained_base_classifiers.pkl"), "rb") as fh:
            self.clfs = pickle.load(fh)
        for name, c in self.clfs.items():
            if name == "XGBoost":
                c.set_params(n_jobs=1)
            elif hasattr(c, "n_jobs"):
                c.n_jobs = 1
        self.fpos = {g.upper(): i for i, g in enumerate(self.features)}

    # ---------- input ----------
    @staticmethod
    def read_table(raw: bytes):
        """Samples as rows, genes as columns. A table with genes as rows is turned around."""
        txt = raw.decode("utf-8-sig", errors="ignore")
        best = None
        for sep in (",", "\t", ";"):
            try:
                df = pd.read_csv(io.StringIO(txt), sep=sep, index_col=0)
            except Exception:
                continue
            if df.shape[1] >= 1 and (best is None or df.shape[1] > best.shape[1]):
                best = df
        if best is None or best.size == 0:
            raise ValueError("The file could not be read as CSV or TSV.")
        df = best
        df.index = df.index.astype(str).str.strip()
        df.columns = df.columns.astype(str).str.strip()
        return df

    def orient(self, df):
        col_hits = sum(c.upper() in self.fpos for c in df.columns)
        row_hits = sum(r.upper() in self.fpos for r in df.index)
        if row_hits > col_hits:
            df = df.T
        df = df.apply(pd.to_numeric, errors="coerce")
        df.columns = df.columns.str.upper()
        df = df.loc[:, ~df.columns.duplicated()]
        hits = sum(c in self.fpos for c in df.columns)
        if hits == 0:
            raise ValueError("No TCUP genes were found. Use HGNC gene symbols as column or row names.")
        return df

    # ---------- prediction ----------
    def _embed_probs(self, Xs):
        emb = np.hstack([_snn(self.nets, Xs), _cae(self.nets, Xs)]).astype(np.float32)
        feats, base_top = [], {}
        n = len(self.model_labels)
        for name, c in self.clfs.items():
            out = np.zeros((len(Xs), n), dtype=np.float32)
            out[:, c.seen_classes_] = c.predict_proba(emb)
            feats.append(out)
            base_top[name] = out[:, self.keep]
        p = _meta(self.nets, np.hstack(feats))[:, self.keep]
        return p / p.sum(axis=1, keepdims=True), base_top

    def predict(self, sample_id, row: pd.Series, sample_type="cancer"):
        ref_key = "healthy" if sample_type == "healthy" else "cancer"
        med = self.ref[f"med_{ref_key}"]
        vals = row.dropna()
        warnings = []
        tpm = vals[vals.index.isin(self.fpos.keys())]
        if len(vals) and float(vals.max()) < 30 and float(vals.min()) >= 0:
            warnings.append("Values look log-transformed (all below 30). TCUP expects TPM, not log2 values.")
        if len(vals) and float(vals.min()) < 0:
            warnings.append("Negative values found. TCUP expects TPM, so these were set to 0.")
        if len(vals) > 5000:
            total = float(vals.clip(lower=0).sum())
            if not 2e5 < total < 2e6:
                warnings.append(f"Values sum to {total:,.0f} across genes; TPM normally sums to about 1,000,000. Check the units.")
        x = med.copy()
        present = np.zeros(len(self.features), bool)
        for g, v in tpm.items():
            i = self.fpos[g]
            x[i] = math.log2(max(float(v), 0.0) + 1.0)
            present[i] = True
        missing = [self.features[i] for i in np.where(~present)[0]]
        if len(missing) > 0.5 * len(self.features):
            warnings.append(f"{len(missing):,} of {len(self.features):,} model genes are missing and were filled with medians. This prediction is unreliable.")
        Xs = self.scaler.transform(x[None, :]).astype(np.float32)
        probs, base_out = self._embed_probs(Xs)
        probs = probs[0]
        order = np.argsort(-probs)
        top = self.labels[order[0]]
        info = self.class_info[top]

        gene_ref = "healthy" if info["source"] == "Healthy" else "cancer"
        gset_name = top if top in self.genes["by_class"] else ("metastatic" if info["source"] == "Metastatic" else "overall")
        gset = self.genes["by_class"][top] if top in self.genes["by_class"] else self.genes[gset_name]
        sig = set(gset["significant"])
        rows = []
        for g in gset["top"]:
            i = self.features.index(g["gene"])
            m, sd = float(self.ref[f"med_{gene_ref}"][i]), float(self.ref[f"sd_{gene_ref}"][i]) or 1e-9
            z = (float(x[i]) - m) / sd
            p = math.erfc(abs(z) / math.sqrt(2))
            rows.append({**g, "significant_in_model": g["gene"] in sig, "expr_log2": round(float(x[i]), 3), "ref_median_log2": round(m, 3),
                         "sigma": round(z, 2), "p": p, "imputed": not bool(present[i])})
        q = _bh([r["p"] for r in rows])
        for r, qq in zip(rows, q):
            r["q"] = float(qq)

        agree = {name: self.labels[int(np.argmax(o[0]))] for name, o in base_out.items()}
        return {
            "sample_id": str(sample_id),
            "prediction": {**info, "probability": float(probs[order[0]])},
            "probabilities": [{"label": self.labels[k], "name": self.class_info[self.labels[k]]["name"], "source": self.class_info[self.labels[k]]["source"],
                               "code": self.class_info[self.labels[k]]["code"], "p": float(probs[k])} for k in order],
            "genes": {"set": gset_name, "set_samples": gset["samples"], "n_significant": len(sig), "reference": gene_ref, "rows": rows},
            "missing": {"count": len(missing), "of": len(self.features), "significant_missing": sorted(set(missing) & sig), "examples": missing[:50]},
            "base_classifiers": agree,
            "warnings": warnings,
            "imputation_reference": ref_key,
        }
