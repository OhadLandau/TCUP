#!/usr/bin/env python3
# DashAppTCUP.py  –  22-May-2025  (visual-tweak update)
# --------------------------------------------------------------------
#  • RadioItems now rendered inline (side-by-side).
#  • “Median values filled” line always shown (even if zero genes).
#  • All other functionality unchanged.
# --------------------------------------------------------------------

import base64, io, pickle, warnings
from pathlib import Path
from typing import List, Tuple, Union

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
from dash import dcc, html, Input, Output, State, no_update, callback

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent

# ───────────────── artefacts ───────────────────────────────────────
# (same helper that builds Cancer/Healthy median pickles if absent)
def _ensure_median_pkls():
    cancer_pkl   = ROOT / "Median_Gene_Values_Cancer.pkl"
    healthy_pkl  = ROOT / "Median_Gene_Values_Healthy.pkl"
    if cancer_pkl.exists() and healthy_pkl.exists():
        return
    merged_pth = ROOT / "merged_data_withStandardScaler.pkl"
    if not merged_pth.exists():
        raise FileNotFoundError("Need merged_data_withStandardScaler.pkl "
                                "to build median files.")
    with open(merged_pth, "rb") as fh:
        merged_obj = pickle.load(fh)
    df_all = pd.concat([
        merged_obj['train_data_scaled'],
        merged_obj['val_data_scaled'],
        merged_obj['test_data_scaled']
    ], ignore_index=True)
    df_all.columns = df_all.columns.str.upper()
    cancer_df  = df_all[df_all['SOURCE'].isin(['TCGA', 'METASTATIC'])]
    healthy_df = df_all[df_all['SOURCE'] == 'GTEX']
    with open(cancer_pkl,  "wb") as fh: pickle.dump(cancer_df.median(numeric_only=True).to_dict(),  fh)
    with open(healthy_pkl, "wb") as fh: pickle.dump(healthy_df.median(numeric_only=True).to_dict(), fh)

_ensure_median_pkls()

with open(ROOT / "Median_Gene_Values_Cancer.pkl", "rb") as fh:
    MEDIANS_CANCER = pickle.load(fh)
with open(ROOT / "Median_Gene_Values_Healthy.pkl", "rb") as fh:
    MEDIANS_HEALTHY = pickle.load(fh)

IMP_DF = pd.read_csv(ROOT / "monte_carlo_gene_importance_averaged.csv").set_index("Gene")

with open(ROOT / "feature_sets.pkl", "rb") as fh:
    FEATURE_COLS: List[str] = pickle.load(fh)["FullLasso"]
with open(ROOT / "BalancedMetastaticStandardScaler.pkl", "rb") as fh:
    SCALER = pickle.load(fh)
with open(ROOT / "trained_base_classifiers.pkl", "rb") as fh:
    BASE_CLS = pickle.load(fh)
with open(ROOT / "merged_data_preprocessed.pkl", "rb") as fh:
    LAB_ENCODER = pickle.load(fh)["label_encoder"]
META_CLASSES = LAB_ENCODER.classes_

_METRIC_CSV = pd.read_csv(ROOT / "test_split_metrics.csv")
_ACC_MAP = {r["Set"].upper(): float(r["Accuracy"]) for _, r in _METRIC_CSV.iterrows()}

import tensorflow as tf
from tensorflow.keras import backend as K      # noqa: F401
tf.get_logger().setLevel("ERROR")
SNN_FULL    = tf.keras.models.load_model(ROOT / "snn_model.h5", compile=False)
SNN_ENCODER = SNN_FULL.layers[2]
CAE_ENCODER = tf.keras.models.load_model(ROOT / "cae_encoder.h5", compile=False)
META_NET    = tf.keras.models.load_model(ROOT / "best_meta_learner_8.h5", compile=False)

# ───────────────── helpers ─────────────────────────────────────────
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
    raw = base64.b64decode(contents.split(",", 1)[1])
    return _robust_read(raw)

def run_pipeline(row: pd.Series, med_dict: dict) -> Tuple[np.ndarray, List[str], dict]:
    vec_log2 = {}
    for g, v in row.items():
        try: vec_log2[g.upper()] = np.log2(float(v) + 1.0)
        except Exception:        pass
    missing = [g for g in FEATURE_COLS if g not in vec_log2]
    for g in missing: vec_log2[g] = med_dict[g]

    X_scaled = SCALER.transform(np.array([vec_log2[g] for g in FEATURE_COLS]).reshape(1,-1))
    emb = np.concatenate([SNN_ENCODER.predict(X_scaled,0), CAE_ENCODER.predict(X_scaled,0)], axis=1)
    meta_feats = np.concatenate([clf.predict_proba(emb) for clf in BASE_CLS.values()], axis=1)
    probs = META_NET.predict(meta_feats,0)[0]
    return probs, missing, vec_log2

def make_bar(prob):
    import plotly.graph_objects as go
    order = prob.argsort()[::-1]
    fig = go.Figure(go.Bar(x=META_CLASSES[order], y=prob[order], marker_color="black"))
    fig.update_layout(showlegend=False,xaxis_tickangle=-90,yaxis_title="Probability",
                      title="Predicted class probabilities",margin=dict(b=160))
    return fig

def acc_for_label(label:str)->str:
    key = label.upper()
    if key in _ACC_MAP: return f"{_ACC_MAP[key]*100:.1f}%"
    base = key.split("_")[0]
    for k,v in _ACC_MAP.items():
        if k.startswith(base): return f"{v*100:.1f}%"
    return "n/a"

def build_summary(label:str,prob:float,missing_len:int):
    acc_str = acc_for_label(label)
    return html.Div([
        html.P([
            "Predicted tissue – ", html.Strong(label), ", ",
            "Probability – ", html.Strong(f"{prob:.2f}"), ", ",
            "TCUP accuracy ≈ ", html.Strong(acc_str)
        ]),
        html.Small(f"Used median values for {missing_len} missing gene(s).")
    ])

def top_gene_list(pred_label:str,row_log2:dict,med_dict:dict)->html.Div:
    col = f"AccuracyDrop_{pred_label}"
    if col not in IMP_DF.columns: col = "AccuracyDrop"
    top20 = IMP_DF[col].sort_values(False).head(20).index
    spans=[]
    for g in top20:
        expr=row_log2.get(g,med_dict[g])
        arrow="▲" if expr>med_dict[g] else "▼"
        cls="up" if arrow=="▲" else "down"
        spans.append(html.Span([g,html.Span(f" {arrow}",className=cls)]))
    return html.Div(spans,id="gene-list")

# ───────────────── UI ─────────────────────────────────────────────
intro_panel = html.Div(id="intro-panel",children=[
    html.Img(src="/assets/logo.png",id="intro-logo"),
    html.Button("Analyze",id="begin-btn",n_clicks=0)
])

upload_comp = dcc.Upload(id="upload",className="dash-upload-area",
                         children=html.Div("Drag & Drop or Browse"),multiple=False)

upload_card = html.Div(id="upload-card",style={"display":"none"},children=[
    html.Img(src="/assets/logo.png",id="logo-top"),
    upload_comp,
    dcc.RadioItems(
        id="sample-type",inline=True,       # ← side-by-side
        options=[{"label":"Cancer","value":"cancer"},
                 {"label":"Healthy","value":"healthy"}],
        value="cancer"
    ),
    dcc.Dropdown(id="sample-select",placeholder="choose sample…",
                 style={"display":"none"}),
    html.Button("Run Analysis",id="analyze-btn",
                className="disabled",disabled=True),
    html.Div(id="status-msg"),
    html.Small("CSV/TSV – first column = sample ID, rest = gene symbols.",
               style={"display":"block","marginTop":"8px"})
])

layout_landing = html.Div(id="landing-panel",
    children=[intro_panel,upload_card,dcc.Store(id="store-df")])

layout_results = html.Div(id="results-panel",style={"display":"none"},children=[
    dcc.Graph(id="prob-graph"),
    html.Div(id="summary-box"),
    html.Div(id="gene-list"),
    html.Button("← Back",id="back-btn")
])

app = dash.Dash(__name__,external_stylesheets=[dbc.themes.BOOTSTRAP],
                suppress_callback_exceptions=True)
app.layout = html.Div([layout_landing,layout_results])
app.server.config["MAX_CONTENT_LENGTH"]=50*1024*1024

# ───────── intro → uploader ───────────────────────────────────────
@callback(Output("intro-panel","style",allow_duplicate=True),
          Output("upload-card","style"),
          Input("begin-btn","n_clicks"),prevent_initial_call=True)
def reveal_uploader(n): return ({"display":"none"},{"display":"block"}) if n else dash.no_update

# ───────── upload handling ───────────────────────────────────────
@callback(Output("store-df","data"),Output("status-msg","children"),
          Output("sample-select","options"),Output("sample-select","value"),
          Output("sample-select","style"),Output("analyze-btn","disabled"),
          Input("upload","contents"),State("upload","filename"),
          prevent_initial_call=True)
def handle_upload(contents,fname):
    if not contents: raise dash.exceptions.PreventUpdate
    df=parse_upload(contents)
    if df is None:
        return no_update,dbc.Alert("❌ Could not read file.",color="danger"),[],None,\
               {"display":"none"},True
    n=len(df)
    alert=dbc.Alert(f"✅ Loaded **{fname}** – {n} sample(s).",color="success",className="mt-2")
    if n==1:
        return df.to_json(date_format="iso",orient="split"),alert,[],None,{"display":"none"},False
    opts=[{"label":str(idx),"value":int(i)} for i,idx in enumerate(df.index)]
    return df.to_json(date_format="iso",orient="split"),alert,opts,None,{"display":"block"},True

@callback(Output("analyze-btn","disabled",allow_duplicate=True),
          Input("sample-select","value"),prevent_initial_call=True)
def enable_analyze(v): return False if v is not None else dash.no_update

app.clientside_callback("return window.dash_clientside.app.flipAnalyze(disabled);",
                        Output("analyze-btn","className"),
                        Input("analyze-btn","disabled"))

# ───────── prediction ────────────────────────────────────────────
@callback(Output("prob-graph","figure"),Output("summary-box","children"),
          Output("gene-list","children"),Output("landing-panel","style"),
          Output("results-panel","style"),
          Input("analyze-btn","n_clicks"),State("store-df","data"),
          State("sample-select","value"),State("sample-type","value"),
          prevent_initial_call=True)
def run_prediction(_,json_df,sample_idx,sample_type):
    df=pd.read_json(json_df,orient="split")
    row=df.iloc[sample_idx or 0]
    med=MEDIANS_CANCER if sample_type=="cancer" else MEDIANS_HEALTHY
    probs,missing,row_log2=run_pipeline(row,med)
    fig=make_bar(probs)
    label=META_CLASSES[probs.argmax()]
    summary=build_summary(label,float(probs.max()),len(missing))
    genes=top_gene_list(label,row_log2,med)
    return fig,summary,genes,{"display":"none"},{"display":"block"}

@callback(Output("landing-panel","style",allow_duplicate=True),
          Output("results-panel","style",allow_duplicate=True),
          Input("back-btn","n_clicks"),prevent_initial_call=True)
def back(_): return {"display":"block"},{"display":"none"}

if __name__=="__main__":
    app.run_server(debug=False)
