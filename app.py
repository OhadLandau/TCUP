#!/usr/bin/env python3
# DashAppTCUP.py – 24 May 2025  (rev-7 full script)
# ------------------------------------------------------------------
# • Exposes `server` for Gunicorn
# • Lazy-loads TensorFlow models to avoid OOM on Render free tier
# • All helper functions INCLUDED – nothing is omitted
# ------------------------------------------------------------------

import base64, io, math, os, pickle, warnings
from functools import lru_cache
from pathlib import Path
from typing import List, Tuple, Union

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
from dash import dcc, html, Input, Output, State, no_update, callback

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent

# ─────────────────── pre-computed artefacts ──────────────────────
with open(ROOT / "Median_Gene_Values_Cancer.pkl",  "rb") as fh: MED_CANCER  = pickle.load(fh)
with open(ROOT / "Std_Gene_Values_Cancer.pkl",     "rb") as fh: STD_CANCER  = pickle.load(fh)
with open(ROOT / "Median_Gene_Values_Healthy.pkl", "rb") as fh: MED_HEALTH = pickle.load(fh)
with open(ROOT / "Std_Gene_Values_Healthy.pkl",    "rb") as fh: STD_HEALTH = pickle.load(fh)

IMP_DF = pd.read_csv(ROOT / "monte_carlo_gene_importance_averaged.csv").set_index("Gene")

with open(ROOT / "feature_sets.pkl", "rb") as fh: FEATURE_COLS = pickle.load(fh)["FullLasso"]
with open(ROOT / "BalancedMetastaticStandardScaler.pkl", "rb") as fh: SCALER   = pickle.load(fh)
with open(ROOT / "trained_base_classifiers.pkl",        "rb") as fh: BASE_CLS = pickle.load(fh)
with open(ROOT / "label_encoder.pkl",                   "rb") as fh: LAB_ENCODER = pickle.load(fh)
META_CLASSES = LAB_ENCODER.classes_

_METRIC_CSV = pd.read_csv(ROOT / "test_split_metrics.csv")
_ACC_MAP = {r["Set"].upper(): float(r["Accuracy"]) for _, r in _METRIC_CSV.iterrows()}

# ─────────────────── lazy-loaded TF models ───────────────────────
@lru_cache(maxsize=1)
def _load_models():
    """Load big Keras models once (first click)."""
    import tensorflow as tf
    from tensorflow.keras import backend as K
    tf.get_logger().setLevel("ERROR")
    custom = {"K": K}

    snn_full = tf.keras.models.load_model(ROOT / "snn_model.h5", compile=False, custom_objects=custom)
    cae_enc  = tf.keras.models.load_model(ROOT / "cae_encoder.h5", compile=False, custom_objects=custom)
    meta_net = tf.keras.models.load_model(ROOT / "best_meta_learner_8.h5", compile=False, custom_objects=custom)
    return snn_full.layers[2], cae_enc, meta_net

# ───────────────────────── helpers ───────────────────────────────
SQRT2 = math.sqrt(2.0)

def _z_two_tail_p(z: float) -> float:
    cdf = 0.5 * (1.0 + math.erf(z / SQRT2))
    return 2.0 * (1.0 - cdf)

def _p_to_stars(p: float) -> str:
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""

def _robust_read(raw: bytes) -> Union[pd.DataFrame, None]:
    txt = raw.decode("utf-8", errors="ignore")
    for sep in (",", "\t", ";", None):
        try:
            df = pd.read_csv(io.StringIO(txt), sep=sep, engine="python")
            if df.shape[1] < 3:
                continue
            df = df.set_index(df.columns[0])
            df.index   = df.index.astype(str)
            df.columns = df.columns.str.strip().str.upper()
            return df
        except Exception:
            continue
    return None

def parse_upload(contents: str):
    return _robust_read(base64.b64decode(contents.split(",", 1)[1]))

def run_pipeline(row: pd.Series, med_dict: dict, std_dict: dict
) -> Tuple[np.ndarray, List[str], dict]:
    SNN_ENCODER, CAE_ENCODER, META_NET = _load_models()

    vec_log2 = {g.upper(): np.log2(float(v) + 1.0)
                for g, v in row.items() if pd.notna(v)}
    missing = [g for g in FEATURE_COLS if g not in vec_log2]
    for g in missing:
        vec_log2[g] = med_dict[g]

    X_scaled = SCALER.transform([[vec_log2[g] for g in FEATURE_COLS]])
    emb = np.concatenate(
        [SNN_ENCODER.predict(X_scaled, verbose=0),
         CAE_ENCODER.predict(X_scaled, verbose=0)], axis=1)
    meta_feats = np.concatenate(
        [clf.predict_proba(emb) for clf in BASE_CLS.values()], axis=1)
    probs = META_NET.predict(meta_feats, verbose=0)[0]
    return probs, missing, vec_log2

def make_bar(prob):
    import plotly.graph_objects as go
    order = prob.argsort()[::-1]
    fig = go.Figure(go.Bar(x=META_CLASSES[order], y=prob[order], marker_color="black"))
    fig.update_layout(showlegend=False, xaxis_tickangle=-90,
                      yaxis_title="Probability",
                      title="Predicted class probabilities",
                      margin=dict(b=160))
    return fig

def acc_for_label(label: str) -> str:
    key = label.upper()
    if key in _ACC_MAP: return f"{_ACC_MAP[key]*100:.1f}%"
    base = key.split("_")[0]
    for k, v in _ACC_MAP.items():
        if k.startswith(base): return f"{v*100:.1f}%"
    return "n/a"

def build_summary(label: str, prob: float,
                  missing: List[str], top128: set) -> html.Div:
    acc_str = acc_for_label(label)
    miss_num = html.Span(str(len(missing)), className="redNum" if missing else "")
    base_small = html.Small(["Used median values for ", miss_num, " missing gene(s)."])

    extra_lines: List[html.Small] = []
    if missing:
        critical = [g for g in missing if g in top128]
        if critical:
            ncrit = len(critical); verb = "was" if ncrit == 1 else "were"
            extra_lines.append(
                html.Small(
                    ["⚠️ **", str(ncrit), f"** gene(s) {verb} imputed from the "
                     "most significant accuracy-affecting gene set for this "
                     "class (top 128; p < 0.05)."],
                    className="warn"
                )
            )
            extra_lines.append(
                html.Small(
                    [str(ncrit), " significant gene(s) imputed: ",
                     ", ".join(sorted(critical))],
                    className="warn"
                )
            )
        else:
            extra_lines.append(
                html.Small("**None** of the imputed genes belong to the top 128 (p < 0.05).",
                           className="ok")
            )
    else:
        extra_lines.append(html.Small("No genes required median imputation.", className="ok"))

    return html.Div([
        html.P(["Predicted tissue – ", html.Strong(label),
                ", Probability – ",   html.Strong(f"{prob:.2f}"),
                ", TCUP accuracy ≈ ", html.Strong(acc_str)]),
        base_small, *extra_lines
    ])

def top_gene_list(pred_label: str, row_log2: dict,
                  med_dict: dict, std_dict: dict
) -> Tuple[List[html.Span], html.Small]:
    col_name = f"AccuracyDrop_{pred_label}"
    if col_name not in IMP_DF.columns: col_name = "AccuracyDrop"

    top20 = IMP_DF[col_name].sort_values(ascending=False).head(20).index
    spans: List[html.Span] = []
    for g in top20:
        expr, median = row_log2.get(g, med_dict[g]), med_dict[g]
        std = std_dict.get(g, 0) or 1e-9
        diff_sigma = (expr - median) / std
        arrow, cls = ("▲", "up") if diff_sigma > 0 else ("▼", "down")
        pval, stars = _z_two_tail_p(abs(diff_sigma)), _p_to_stars(_z_two_tail_p(abs(diff_sigma)))
        spans.append(html.Span([
            g, html.Span(f" {arrow}", className=cls),
            html.Span(f" ({diff_sigma:+.1f}σ){stars}", className="std")
        ]))

    legend = html.Small("* p < 0.05 ** p < 0.01 *** p < 0.001 "
                        "(two-tailed z-test of ±σ across all samples per gene).")
    return spans, legend

# ─────────────────────────── UI ──────────────────────────────────
intro_panel = html.Div(id="intro-panel", children=[
    html.Img(src="/assets/logo.png", id="intro-logo"),
    html.Button("Analyze", id="begin-btn", n_clicks=0),
])

upload_comp = dcc.Upload(id="upload", className="dash-upload-area",
                         children=html.Div("Drag & Drop or Browse"), multiple=False)

upload_card = html.Div(id="upload-card", style={"display": "none"}, children=[
    html.Img(src="/assets/logo.png", id="logo-top"),
    upload_comp,
    dcc.RadioItems(id="sample-type", inline=True, value="cancer",
                   options=[{"label": "Cancer",  "value": "cancer"},
                            {"label": "Healthy", "value": "healthy"}]),
    dcc.Dropdown(id="sample-select", placeholder="choose sample…",
                 style={"display": "none"}),
    html.Button("Run Analysis", id="analyze-btn", className="disabled", disabled=True),
    html.Div(id="status-msg"),
    html.Small("CSV/TSV – first column = sample ID, the rest = gene symbols.",
               style={"display": "block", "marginTop": "8px"}),
])

layout_landing = html.Div(id="landing-panel",
                          children=[intro_panel, upload_card, dcc.Store(id="store-df")])
layout_results = html.Div(id="results-panel", style={"display": "none"}, children=[
    dcc.Graph(id="prob-graph"),
    html.Div(id="summary-box"),
    html.Div(id="gene-list"),
    html.Div(id="gene-note"),
    html.Button("← Back", id="back-btn"),
])

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],
                suppress_callback_exceptions=True)
server = app.server                      # Gunicorn entry-point
app.layout = html.Div([layout_landing, layout_results])
app.server.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024

# ───────────────────────── callbacks ─────────────────────────────
@callback(Output("intro-panel", "style", allow_duplicate=True),
          Output("upload-card", "style"),
          Input("begin-btn", "n_clicks"), prevent_initial_call=True)
def reveal_uploader(n):
    if n: return {"display": "none"}, {"display": "block"}
    raise dash.exceptions.PreventUpdate

@callback(Output("store-df", "data"), Output("status-msg", "children"),
          Output("sample-select", "options"), Output("sample-select", "value"),
          Output("sample-select", "style"),  Output("analyze-btn", "disabled"),
          Input("upload", "contents"), State("upload", "filename"),
          prevent_initial_call=True)
def handle_upload(contents, fname):
    if not contents: raise dash.exceptions.PreventUpdate
    df = parse_upload(contents)
    if df is None:
        return no_update, dbc.Alert("❌ Could not read file.", color="danger"), \
               [], None, {"display": "none"}, True
    n = len(df)
    alert = dbc.Alert(f"✅ Loaded **{fname}** – {n} sample(s).",
                      color="success", className="mt-2")
    if n == 1:
        return df.to_json(date_format="iso", orient="split"), alert, \
               [], None, {"display": "none"}, False
    opts = [{"label": str(idx), "value": int(i)} for i, idx in enumerate(df.index)]
    return df.to_json(date_format="iso", orient="split"), alert, \
           opts, None, {"display": "block"}, True

@callback(Output("analyze-btn", "disabled", allow_duplicate=True),
          Input("sample-select", "value"), prevent_initial_call=True)
def enable_analyze(val):
    if val is None: raise dash.exceptions.PreventUpdate
    return False

app.clientside_callback(
    "return window.dash_clientside.app.flipAnalyze(disabled);",
    Output("analyze-btn", "className"), Input("analyze-btn", "disabled"))

@callback(Output("prob-graph", "figure"), Output("summary-box", "children"),
          Output("gene-list", "children"), Output("gene-note", "children"),
          Output("landing-panel", "style"), Output("results-panel", "style"),
          Input("analyze-btn", "n_clicks"), State("store-df", "data"),
          State("sample-select", "value"), State("sample-type", "value"),
          prevent_initial_call=True)
def run_prediction(_, json_df, sample_idx, sample_type):
    df   = pd.read_json(json_df, orient="split")
    row  = df.iloc[sample_idx or 0]
    med_dict = MED_CANCER if sample_type == "cancer" else MED_HEALTH
    std_dict = STD_CANCER if sample_type == "cancer" else STD_HEALTH

    probs, missing, row_log2 = run_pipeline(row, med_dict, std_dict)
    label = META_CLASSES[probs.argmax()]

    col = f"AccuracyDrop_{label}"
    if col not in IMP_DF.columns: col = "AccuracyDrop"
    sig_top128 = (IMP_DF.loc[IMP_DF["p_value"] < 0.05, col]
                  .sort_values(ascending=False).head(128).index)

    summary = build_summary(label, float(probs.max()), missing, set(sig_top128))
    spans, legend = top_gene_list(label, row_log2, med_dict, std_dict)
    return make_bar(probs), summary, spans, legend, \
           {"display": "none"}, {"display": "block"}

@callback(Output("landing-panel", "style", allow_duplicate=True),
          Output("results-panel", "style", allow_duplicate=True),
          Input("back-btn", "n_clicks"), prevent_initial_call=True)
def go_back(_): return {"display": "block"}, {"display": "none"}

# ───────────────────────── run local dev ─────────────────────────
if __name__ == "__main__":
    os.environ.setdefault("WEB_CONCURRENCY", "1")   # single worker
    app.run_server(debug=True)
