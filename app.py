"""
Shopper Intent | Model Evaluation Workspace
Online Shoppers Purchasing Intention - M.Tech AIML, Assignment 2

Structure
    1. Configuration & constants
    2. Visual system (CSS)
    3. Model loading & data validation
    4. Evaluation logic
    5. Charts
    6. UI components (cards, HTML tables)
    7. Page sections
    8. Main
"""

from html import escape
from pathlib import Path
import hashlib
import io

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# Best effort only. The reliable way to get the light theme is
# `streamlit run app.py --theme.base light` or .streamlit/config.toml.
# The app itself no longer depends on the theme: charts, tables, tabs,
# buttons and inputs are all styled explicitly below.
try:
    st._config.set_option("theme.base", "light")
except Exception:
    pass


# =====================================================================
# 1. Configuration & constants
# =====================================================================

st.set_page_config(
    page_title="Shopper Intent | Evaluation Studio",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="collapsed",
)

TARGET = "Revenue"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "knn.joblib",
    "Gaussian Naive Bayes": "gaussian_naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}

MODEL_COLORS = {
    "Logistic Regression": "#0E6B57",
    "Decision Tree": "#C9A24B",
    "K-Nearest Neighbors": "#2F5D9E",
    "Gaussian Naive Bayes": "#7A4A8F",
    "Support Vector Machine": "#B03A48",
}

NUMERIC_FEATURES = [
    "Administrative", "Administrative_Duration",
    "Informational", "Informational_Duration",
    "ProductRelated", "ProductRelated_Duration",
    "BounceRates", "ExitRates", "PageValues", "SpecialDay",
]

CATEGORICAL_FEATURES = [
    "Month", "OperatingSystems", "Browser", "Region",
    "TrafficType", "VisitorType", "Weekend",
]

METRICS = ["Accuracy", "AUC", "Precision", "Recall", "F1 Score", "MCC"]

METRIC_NOTES = {
    "Accuracy": "Overall correct predictions",
    "AUC": "Ranking quality",
    "Precision": "Predicted buyers who bought",
    "Recall": "Real buyers found",
    "F1 Score": "Precision and recall balance",
    "MCC": "Imbalance-safe score",
}

FEATURE_DESCRIPTIONS = {
    "Administrative": "Number of administrative pages visited.",
    "Administrative_Duration": "Time spent on administrative pages, in seconds.",
    "Informational": "Number of informational pages visited.",
    "Informational_Duration": "Time spent on informational pages, in seconds.",
    "ProductRelated": "Number of product-related pages visited.",
    "ProductRelated_Duration": "Time spent on product-related pages, in seconds.",
    "BounceRates": "Average bounce rate for the pages visited.",
    "ExitRates": "Average exit rate for the pages visited.",
    "PageValues": "Average page value associated with the session.",
    "SpecialDay": "Closeness of the visit to a special day.",
    "Month": "Month in which the session occurred.",
    "OperatingSystems": "Operating system category.",
    "Browser": "Browser category.",
    "Region": "Geographic region category.",
    "TrafficType": "Traffic source category.",
    "VisitorType": "New, returning, or other visitor category.",
    "Weekend": "Whether the session occurred on a weekend.",
}

MODEL_USAGE = {
    "Logistic Regression": "Useful as an interpretable linear baseline. It estimates purchase likelihood from a weighted combination of session features.",
    "Decision Tree": "Useful when rule-like decision paths are important. It can capture non-linear feature interactions but may overfit without depth controls.",
    "K-Nearest Neighbors": "Useful for similarity-based classification. It predicts from nearby training examples and can be sensitive to feature scaling and the choice of neighbours.",
    "Gaussian Naive Bayes": "A fast probabilistic baseline. It assumes conditional feature independence and Gaussian feature distributions, assumptions that may not fit this dataset well.",
    "Support Vector Machine": "Useful for learning a separating boundary in feature space. Probability estimates can be more computationally expensive.",
}

MODEL_DESCRIPTIONS = {
    "Logistic Regression": "Linear classification model used as a baseline.",
    "Decision Tree": "Rule-based model that learns decision splits.",
    "K-Nearest Neighbors": "Classifies a session using nearby training examples.",
    "Gaussian Naive Bayes": "Probabilistic classifier based on feature likelihoods.",
    "Support Vector Machine": "Classifier that learns a maximum-margin boundary.",
}

MODE_ALL = "Compare all five"
MODE_ONE = "Single model"

APP_DIR = Path(__file__).resolve().parent
MODEL_DIR_CANDIDATES = [
    APP_DIR / "models",
    APP_DIR / "src" / "models",
    APP_DIR.parent / "models",
]


def locate_model_dir():
    for directory in MODEL_DIR_CANDIDATES:
        if all((directory / f).is_file() for f in MODEL_FILES.values()):
            return directory
    return next((d for d in MODEL_DIR_CANDIDATES if d.exists()), MODEL_DIR_CANDIDATES[0])


MODEL_DIR = locate_model_dir()


# =====================================================================
# 2. Visual system
# =====================================================================
# Direction: "pine and brass". A dark pine control rail beside a porcelain
# canvas, with brass as the single accent.
#   Porcelain #F5F6F3   Pine #0F2D2A / #1D4A43   Brass #B08D57
# Type: Fraunces (headings, numerals) + Hanken Grotesk (interface)
#
# Selectors are deliberately broad (data-baseweb / role / key classes) and
# use !important so Streamlit's own theme (red accent, dark text colours)
# cannot leak through.

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
:root {
  color-scheme: light;
  --ink:#30243A; --muted:#716779; --subtle:#9A909F;
  --blue:#A85F45; --blue-dark:#874832; --teal:#477C70;
  --canvas:#F4F0E9; --surface:#FFFDFA; --line:#E7DED3;
  --soft-blue:#F4E7DF; --green:#477C70; --red:#B85C63;
}
html,body,.stApp,[data-testid="stAppViewContainer"],[data-testid="stMain"] {
  font-family:'DM Sans',sans-serif!important; color:var(--ink)!important; color-scheme:light;
}
.stApp {background:var(--canvas)!important;}
[data-testid="stAppViewContainer"],[data-testid="stMain"],[data-testid="stHeader"] {background:transparent!important;}
.block-container {max-width:1500px;padding:1.5rem 2.5rem 2.5rem;}
[data-testid="stSidebar"],[data-testid="stSidebarCollapsedControl"] {display:none!important;}
[data-testid="stToolbar"],[data-testid="stDeployButton"],[data-testid="stAppDeployButton"],.stDeployButton,[data-testid="stDecoration"],#MainMenu,footer {display:none!important;}
[data-testid="stHeader"] {height:0.8rem;}
h1,h2,h3,h4,h5,.brand-name,.sec-title {font-family:'Manrope',sans-serif!important;color:var(--ink)!important;letter-spacing:-.035em;font-weight:800!important;}
p,li,[data-testid="stMarkdownContainer"] {color:var(--ink);}
[data-testid="stCaptionContainer"],.stCaption,[data-testid="stCaptionContainer"] p {color:var(--muted)!important;}
.topbar {display:flex;justify-content:space-between;align-items:center;gap:22px;padding:4px 0 24px;border-bottom:1px solid var(--line);margin-bottom:24px;}
.brand {display:flex;align-items:center;gap:14px;}
.logo {width:46px;height:46px;border-radius:13px;background:linear-gradient(145deg,#A85F45,#874832);color:white;display:grid;place-items:center;font-family:'Manrope',sans-serif;font-weight:800;font-size:21px;box-shadow:0 7px 18px rgba(168,95,69,.18);}
.brand-name {font-size:25px;line-height:1.1;}
.brand-sub {font-size:13px;color:var(--muted);margin-top:4px;}
.pills {display:flex;gap:8px;flex-wrap:wrap;justify-content:flex-end;}
.pill {font-size:12px;font-weight:700;color:#42516B;background:#fff;border:1px solid var(--line);border-radius:8px;padding:7px 11px;}
.pill.brass {background:var(--soft-blue);color:var(--blue);border-color:#EBD6CA;}
.sec-title {font-size:21px;margin:0 0 4px;}
.sec-help {color:var(--muted);font-size:13px;line-height:1.55;margin-bottom:14px;}
div[class*="st-key-card_"],div[class*="st-key-chart_"],[data-testid="stExpander"],[data-testid="stMetric"],.dt-wrap,.kpi {
 background:var(--surface)!important;border:1px solid var(--line)!important;border-radius:14px!important;box-shadow:0 3px 14px rgba(25,43,78,.045)!important;
}
div[class*="st-key-card_"] {padding:22px 24px;}
div[class*="st-key-chart_"] {padding:10px 14px 4px;}
[data-testid="stMetric"] {padding:14px 16px;border-radius:12px!important;}
.kpi {padding:15px 15px 13px;height:100%;}
.kpi-label {font-size:12px;font-weight:700;color:var(--muted);}
.kpi-value {font-family:'Manrope',sans-serif;font-size:29px;font-weight:800;color:var(--ink);margin:5px 0 10px;line-height:1.1;}
.kpi-bar {height:5px;background:#E9EDF5;border-radius:20px;overflow:hidden;}
.kpi-bar>span {display:block;height:100%;background:linear-gradient(90deg,var(--blue),#6B87EE);border-radius:20px;}
.kpi-note {font-size:11px;color:var(--subtle);margin-top:8px;line-height:1.35;}
.callout {border-radius:10px;padding:12px 15px;font-size:13px;line-height:1.55;margin:8px 0 14px;border:1px solid;}
.callout.info {background:#EFF5FF;border-color:#D7E4FF;color:#294B91;}
.callout.best {background:#EAF7F1;border-color:#CBEBDD;color:#176B4F;}
.callout.warn {background:#FFF1F1;border-color:#F4D5D8;color:#963C48;}
.steps {display:grid;gap:17px;margin-top:10px;}
.step {display:flex;gap:13px;align-items:flex-start;}
.step-n {flex:none;width:30px;height:30px;border-radius:9px;background:var(--soft-blue);color:var(--blue);display:grid;place-items:center;font-weight:800;font-size:13px;}
.step b {display:block;font-size:14px;color:var(--ink);margin-bottom:2px;}
.step span {font-size:13px;color:var(--muted);line-height:1.5;}
.dt-wrap {overflow:auto;margin-bottom:8px;}
table.dt {width:100%;border-collapse:separate;border-spacing:0;font-size:13px;color:var(--ink);}
table.dt th {position:sticky;top:0;z-index:1;text-align:left;white-space:nowrap;background:#F5F0E9;color:#45536B;font-weight:700;font-size:12px;padding:12px 14px;border-bottom:1px solid var(--line);}
table.dt td {padding:11px 14px;border-bottom:1px solid #EDF0F5;white-space:nowrap;vertical-align:middle;color:var(--ink);}
table.dt tr:last-child td {border-bottom:none;} table.dt tbody tr:hover td {background:#FBF8F3;}
table.dt tr.top td {background:#F6ECE5;}
.wrap-cell {white-space:normal!important;min-width:280px;line-height:1.5;}
.mdot {display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:9px;vertical-align:middle;}
.bar-cell {display:flex;align-items:center;gap:10px;}
.bar-track {width:68px;height:6px;background:#EAE2D8;border-radius:20px;overflow:hidden;flex:none;}
.bar-fill {display:block;height:100%;background:var(--blue);border-radius:20px;}
.bar-cell b {font-weight:700;font-variant-numeric:tabular-nums;}
.badge {display:inline-block;padding:3px 9px;border-radius:6px;font-size:11px;font-weight:700;}
.badge.ok {background:#E8F6EF;color:#16734F;} .badge.bad {background:#FCEBED;color:#A63E4B;}
.crown {margin-left:8px;font-size:10px;font-weight:800;color:#A85F45;background:#F4E7DF;padding:3px 7px;border-radius:5px;}
[data-testid="stWidgetLabel"] p,[data-testid="stWidgetLabel"] label {color:#34435C!important;font-weight:700!important;font-size:13px!important;}
div[data-baseweb="select"]>div {background:#fff!important;border:1px solid #DED4C9!important;border-radius:9px!important;min-height:42px;}
div[data-baseweb="select"] * {color:var(--ink)!important;} div[data-baseweb="select"]:focus-within>div {border-color:var(--blue)!important;box-shadow:0 0 0 3px rgba(168,95,69,.13)!important;}
div[data-baseweb="popover"] ul,div[role="listbox"] {background:#fff!important;border:1px solid var(--line)!important;border-radius:9px!important;}
div[role="option"],li[role="option"] {background:#fff!important;color:var(--ink)!important;}
div[role="option"]:hover,li[role="option"]:hover {background:#F6ECE5!important;}
div[role="radiogroup"] {background:#F0F3F8;padding:4px;border-radius:10px;gap:4px!important;display:flex!important;flex-direction:row!important;width:100%;}
div[role="radiogroup"] label {flex:1;justify-content:center;background:transparent;padding:8px;border-radius:7px;margin:0!important;cursor:pointer;}
div[role="radiogroup"] label>div:first-of-type {display:none!important;}
div[role="radiogroup"] label p,div[role="radiogroup"] label span {color:#56647B!important;font-weight:700!important;font-size:13px!important;text-align:center;}
div[role="radiogroup"] label:has(input:checked) {background:#fff;box-shadow:0 1px 4px rgba(48,36,58,.10);}
div[role="radiogroup"] label:has(input:checked) p,div[role="radiogroup"] label:has(input:checked) span {color:var(--blue)!important;}
[data-testid="stFileUploader"] section {background:#FBF8F3!important;border:1.5px dashed #CDB9AC!important;border-radius:11px!important;padding:12px!important;}
[data-testid="stFileUploader"] section * {color:var(--ink)!important;}
[data-testid="stFileUploader"] section button {background:#fff!important;border:1px solid var(--line)!important;border-radius:8px!important;}
[data-testid="stFileUploader"] section button * {color:var(--blue)!important;}
.stButton>button,.stDownloadButton>button {border-radius:9px!important;min-height:42px;font-weight:700;background:var(--blue)!important;border:1px solid var(--blue)!important;box-shadow:0 4px 10px rgba(168,95,69,.14);}
.stButton>button *, .stDownloadButton>button * {color:#fff!important;font-weight:700;}
.stButton>button:hover,.stDownloadButton>button:hover {background:var(--blue-dark)!important;border-color:var(--blue-dark)!important;}
.stButton>button:disabled {background:#EEE8E0!important;border-color:#EEE8E0!important;box-shadow:none;}
.stButton>button:disabled * {color:#8B96A8!important;}
[data-baseweb="tab-list"] {gap:5px!important;padding:4px!important;background:#EEE7DE!important;border-radius:10px!important;width:fit-content;}
[data-baseweb="tab-highlight"],[data-baseweb="tab-border"] {display:none!important;}
button[data-baseweb="tab"],button[role="tab"] {background:transparent!important;padding:9px 15px!important;border:0!important;border-radius:7px!important;box-shadow:none!important;}
button[role="tab"] * {color:#58667D!important;font-weight:700!important;font-size:13px!important;}
button[role="tab"][aria-selected="true"] {background:#fff!important;box-shadow:0 1px 4px rgba(48,36,58,.10)!important;}
button[role="tab"][aria-selected="true"] * {color:var(--blue)!important;}
[data-testid="stMetricLabel"],[data-testid="stMetricLabel"] * {color:var(--muted)!important;font-weight:700!important;}
[data-testid="stMetricValue"],[data-testid="stMetricValue"] * {color:var(--ink)!important;font-family:'Manrope',sans-serif!important;font-weight:800!important;}
[data-testid="stDataFrame"] {border:1px solid var(--line);border-radius:10px;}
.file-chip {display:flex;align-items:center;gap:9px;margin:8px 0 12px;background:#EAF7F1;border:1px solid #CBEBDD;color:#176B4F;border-radius:8px;padding:9px 12px;font-size:12px;font-weight:600;}
.dot {width:8px;height:8px;border-radius:50%;background:#1A9B70;flex:none;}
.rail-rule {height:1px;background:var(--line);margin:17px 0;}
.rail-title {font-family:'Manrope',sans-serif;font-size:18px;font-weight:800;color:var(--ink);margin-bottom:4px;}
.rail-help {font-size:12.5px;line-height:1.5;color:var(--muted);margin-bottom:15px;}
.app-footer {border-top:1px solid var(--line);margin-top:34px;padding-top:14px;color:var(--subtle);font-size:12px;display:flex;justify-content:space-between;flex-wrap:wrap;gap:8px;}
@media(max-width:900px){.block-container{padding:1rem 1rem 2rem}.topbar{flex-direction:column;align-items:flex-start}.pills{justify-content:flex-start}.brand-name{font-size:22px;}}

div[data-baseweb="select"] {opacity:1!important;}
div[data-baseweb="select"]>div {background:#FFFDFA!important;border-color:#D8C9BD!important;}
div[data-baseweb="select"] [aria-disabled="true"] {opacity:.58;}
[data-testid="stWidgetLabel"] p {color:#493B50!important;}
</style>
    """,
    unsafe_allow_html=True,
)


# =====================================================================
# 3. Model loading & data validation
# =====================================================================

@st.cache_resource(show_spinner=False)
def load_model(model_path, modified_time):
    return joblib.load(model_path)


def model_signature():
    signature = []
    for name, filename in MODEL_FILES.items():
        path = MODEL_DIR / filename
        signature.append((name, path.stat().st_mtime_ns if path.exists() else None))
    return tuple(signature)


def get_pipeline(model_name):
    path = MODEL_DIR / MODEL_FILES[model_name]
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing model file: {path}. Put all five .joblib files in a 'models' folder."
        )
    return load_model(str(path), path.stat().st_mtime_ns)


def normalize_target(series):
    if series.isna().any():
        raise ValueError("The Revenue target contains missing values.")

    if pd.api.types.is_bool_dtype(series):
        result = series.astype(int)
    else:
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().all() and numeric.isin([0, 1]).all():
            result = numeric.astype(int)
        else:
            text = series.astype(str).str.strip().str.lower()
            result = text.map({"true": 1, "false": 0, "1": 1, "0": 0})
            if result.isna().any():
                invalid = text[result.isna()].unique().tolist()[:5]
                raise ValueError(
                    f"Revenue must contain only 0/1 or True/False. Invalid values: {invalid}"
                )
            result = result.astype(int)

    if result.nunique() != 2:
        raise ValueError(
            "The uploaded test CSV must contain both target classes "
            "(Revenue = 0 and Revenue = 1) to calculate AUC."
        )
    return result


def validate_test_data(frame, reference_pipeline):
    if frame.empty:
        raise ValueError("The uploaded CSV has no rows.")
    if frame.columns.duplicated().any():
        raise ValueError("The CSV contains duplicate column names.")
    if TARGET not in frame.columns:
        raise ValueError("The test CSV must include the Revenue target column.")
    if not hasattr(reference_pipeline, "named_steps"):
        raise ValueError("The saved model is not a preprocessing-and-classifier pipeline.")

    preprocessor = reference_pipeline.named_steps.get("preprocessor")
    if preprocessor is None or not hasattr(preprocessor, "feature_names_in_"):
        raise ValueError(
            "The saved pipeline must contain a fitted preprocessing step named 'preprocessor'."
        )

    expected = list(preprocessor.feature_names_in_)
    actual = [c for c in frame.columns if c != TARGET]
    missing = [c for c in expected if c not in actual]
    extra = [c for c in actual if c not in expected]
    if missing or extra:
        details = []
        if missing:
            details.append("Missing columns: " + ", ".join(missing))
        if extra:
            details.append("Unexpected columns: " + ", ".join(extra))
        raise ValueError(" | ".join(details))

    X = frame[expected].copy()
    y = normalize_target(frame[TARGET])

    invalid_values = {}
    for column in NUMERIC_FEATURES:
        if column in X.columns:
            original = X[column]
            converted = pd.to_numeric(original, errors="coerce")
            bad = original.notna() & converted.isna()
            if bad.any():
                invalid_values[column] = int(bad.sum())
            X[column] = converted
    if invalid_values:
        details = ", ".join(f"{c}: {n}" for c, n in invalid_values.items())
        raise ValueError(
            "Some numeric features contain non-numeric values. Correct them in the CSV: " + details
        )

    for column in CATEGORICAL_FEATURES:
        if column in X.columns:
            X[column] = X[column].astype("object")

    return X, y


# =====================================================================
# 4. Evaluation logic
# =====================================================================

def evaluate_one(pipeline, X, y):
    predictions = np.asarray(pipeline.predict(X)).astype(int)

    if not hasattr(pipeline, "predict_proba"):
        raise ValueError("A saved classifier does not provide predict_proba(), required for AUC.")

    classes = list(pipeline.classes_)
    if 1 not in classes:
        raise ValueError("The saved classifier was not trained with Revenue = 1.")
    probabilities = pipeline.predict_proba(X)[:, classes.index(1)]

    scores = {
        "Accuracy": accuracy_score(y, predictions),
        "AUC": roc_auc_score(y, probabilities),
        "Precision": precision_score(y, predictions, zero_division=0),
        "Recall": recall_score(y, predictions, zero_division=0),
        "F1 Score": f1_score(y, predictions, zero_division=0),
        "MCC": matthews_corrcoef(y, predictions),
    }
    return scores, predictions, probabilities


def evaluate_all_models(csv_bytes, selected_models):
    frame = pd.read_csv(io.BytesIO(csv_bytes))
    reference = get_pipeline(selected_models[0])
    X, y = validate_test_data(frame, reference)

    results, rows = {}, []
    for name in selected_models:
        pipeline = get_pipeline(name)
        preprocessor = pipeline.named_steps.get("preprocessor")
        if preprocessor is None or list(preprocessor.feature_names_in_) != list(X.columns):
            raise ValueError(
                f"{name} was trained with a different feature schema. "
                "Retrain/save the model using the same preprocessing columns."
            )
        scores, predictions, probabilities = evaluate_one(pipeline, X, y)
        results[name] = {"scores": scores, "predictions": predictions, "probabilities": probabilities}
        rows.append({"Model": name, **scores})

    return frame, X, y, results, pd.DataFrame(rows)


# =====================================================================
# 5. Charts (explicit colours, theme=None)
# =====================================================================

INK = "#0A2420"
TEXT = "#35504C"
GRID = "#E9EDEA"
AXIS = "#C9D3CE"
FONT = "Hanken Grotesk, sans-serif"
HEAD = "Fraunces, Georgia, serif"


def style_figure(fig, title, height=360, margin=None, **layout):
    base_layout = {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": dict(family=FONT, color=TEXT, size=12.5),
        "title": dict(text=title, font=dict(family=HEAD, size=17, color=INK), x=0.01, xanchor="left"),
        "height": height,
        "margin": margin or dict(l=50, r=24, t=56, b=46),
        "hoverlabel": dict(bgcolor="#FFFFFF", font=dict(color=INK, family=FONT), bordercolor=AXIS),
        "legend": dict(font=dict(color=TEXT, size=12), bgcolor="rgba(0,0,0,0)"),
    }
    base_layout.update(layout)
    fig.update_layout(**base_layout)
    fig.update_xaxes(gridcolor=GRID, linecolor=AXIS, zerolinecolor=GRID,
                     tickfont=dict(color=TEXT), title_font=dict(color=TEXT))
    fig.update_yaxes(gridcolor=GRID, linecolor=AXIS, zerolinecolor=GRID,
                     tickfont=dict(color=TEXT), title_font=dict(color=TEXT))
    return fig


def show(fig, key):
    with st.container(key=f"chart_{key}"):
        st.plotly_chart(fig, theme=None, width="stretch",
                        config={"displayModeBar": False}, key=f"plot_{key}")


def confusion_figure(y, predictions):
    matrix = confusion_matrix(y, predictions, labels=[0, 1])
    fig = go.Figure(go.Heatmap(
        z=matrix,
        x=["Predicted: no purchase", "Predicted: purchase"],
        y=["Actual: no purchase", "Actual: purchase"],
        colorscale=[[0, "#EAF2EE"], [1, "#0E6B57"]],
        showscale=False, xgap=5, ygap=5,
        hovertemplate="%{y}<br>%{x}<br>Count: %{z:,}<extra></extra>",
    ))
    top = matrix.max() or 1
    for i in range(2):
        for j in range(2):
            fig.add_annotation(
                x=j, y=i, text=f"{matrix[i][j]:,}", showarrow=False,
                font=dict(family=HEAD, size=24,
                          color="#FFFFFF" if matrix[i][j] > top * 0.5 else INK),
            )
    style_figure(fig, "Confusion matrix", yaxis=dict(autorange="reversed"))
    fig.update_xaxes(showgrid=False, linecolor="rgba(0,0,0,0)")
    fig.update_yaxes(showgrid=False, linecolor="rgba(0,0,0,0)")
    return fig


def roc_figure(y, probabilities, auc_value):
    fpr, tpr, _ = roc_curve(y, probabilities)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr, mode="lines", name=f"ROC (AUC {auc_value:.3f})",
        line=dict(color="#0E6B57", width=3),
        fill="tozeroy", fillcolor="rgba(14,107,87,0.13)",
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines", name="Random baseline",
        line=dict(color="#C9A24B", dash="dash", width=1.8),
    ))
    style_figure(fig, "ROC curve", legend=dict(orientation="h", y=-0.24, font=dict(color=TEXT)))
    fig.update_xaxes(title="False positive rate", range=[0, 1])
    fig.update_yaxes(title="True positive rate", range=[0, 1.02])
    return fig


def roc_overlay_figure(y, results):
    fig = go.Figure()
    for name, res in results.items():
        fpr, tpr, _ = roc_curve(y, res["probabilities"])
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines",
            name=f"{name} ({res['scores']['AUC']:.3f})",
            line=dict(color=MODEL_COLORS[name], width=2.6),
        ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines", name="Random",
        line=dict(color="#A9B4B0", dash="dash", width=1.3),
    ))
    style_figure(fig, "ROC curves, all classifiers", height=420,
                 legend=dict(orientation="h", y=-0.3, font=dict(color=TEXT, size=11.5)))
    fig.update_xaxes(title="False positive rate", range=[0, 1])
    fig.update_yaxes(title="True positive rate", range=[0, 1.02])
    return fig


def radar_figure(comparison):
    fig = go.Figure()
    for _, row in comparison.iterrows():
        values = [max(row[m], 0) for m in METRICS]
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]], theta=METRICS + [METRICS[0]],
            name=row["Model"], mode="lines",
            line=dict(color=MODEL_COLORS[row["Model"]], width=2.6),
        ))
    style_figure(
        fig, "Metric profile by classifier", height=420,
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(range=[0, 1], gridcolor=GRID, linecolor=AXIS,
                            tickfont=dict(color=TEXT, size=10)),
            angularaxis=dict(gridcolor=GRID, linecolor=AXIS, tickfont=dict(color=INK, size=12)),
        ),
        legend=dict(orientation="h", y=-0.12, font=dict(color=TEXT, size=11.5)),
    )
    return fig


def metric_bar_figure(comparison, metric):
    ordered = comparison.sort_values(metric, ascending=True)
    fig = go.Figure(go.Bar(
        x=ordered[metric], y=ordered["Model"], orientation="h",
        marker=dict(color=[MODEL_COLORS[m] for m in ordered["Model"]]),
        text=ordered[metric].map(lambda v: f"{v:.3f}"),
        textposition="outside", textfont=dict(color=INK, size=13), cliponaxis=False,
        hovertemplate="%{y}<br>" + metric + ": %{x:.4f}<extra></extra>",
    ))
    style_figure(fig, f"Classifier comparison: {metric}", height=340,
                 margin=dict(l=175, r=70, t=56, b=40), bargap=0.38)
    fig.update_xaxes(range=[min(0, ordered[metric].min() - 0.05), 1.12], title=metric)
    fig.update_yaxes(title="", automargin=True)
    return fig


# =====================================================================
# 6. UI components
# =====================================================================

def section(title, help_text=""):
    html = f'<div class="sec-title">{title}</div>'
    if help_text:
        html += f'<div class="sec-help">{help_text}</div>'
    st.markdown(html, unsafe_allow_html=True)


def callout(text, kind="info"):
    st.markdown(f'<div class="callout {kind}">{text}</div>', unsafe_allow_html=True)


def kpi_card(label, value, note):
    pct = max(0.0, min(1.0, float(value))) * 100
    st.markdown(
        f'<div class="kpi"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value:.4f}</div>'
        f'<div class="kpi-bar"><span style="width:{pct:.1f}%"></span></div>'
        f'<div class="kpi-note">{note}</div></div>',
        unsafe_allow_html=True,
    )


def bar_cell(value, low=0.0, high=1.0):
    pct = max(0.0, min(1.0, (float(value) - low) / (high - low))) * 100
    return (f'<div class="bar-cell"><span class="bar-track"><span class="bar-fill" '
            f'style="width:{pct:.1f}%"></span></span><b>{float(value):.4f}</b></div>')


def model_cell(name):
    color = MODEL_COLORS.get(name, "#8A9794")
    return f'<span class="mdot" style="background:{color}"></span>{escape(str(name))}'


def render_html(html):
    """Raw HTML without the markdown parser (safer for big tables)."""
    if hasattr(st, "html"):
        st.html(html)
    else:
        st.markdown(html, unsafe_allow_html=True)


def html_table(df, renderers=None, float_fmt="{:.4f}", max_height=None, top_row=None, wrap_cols=()):
    """Theme-independent table. `renderers` maps column -> function(value) -> html."""
    renderers = renderers or {}
    head = "".join(f"<th>{escape(str(c))}</th>" for c in df.columns)
    rows = []
    for idx, (_, row) in enumerate(df.iterrows()):
        cells = []
        for col in df.columns:
            value = row[col]
            if col in renderers:
                content = renderers[col](value)
            elif isinstance(value, (float, np.floating)):
                content = float_fmt.format(value)
            else:
                content = escape(str(value))
            cls = ' class="wrap-cell"' if col in wrap_cols else ""
            cells.append(f"<td{cls}>{content}</td>")
        row_cls = ' class="top"' if top_row is not None and idx == top_row else ""
        rows.append(f"<tr{row_cls}>{''.join(cells)}</tr>")
    style = f' style="max-height:{max_height}px"' if max_height else ""
    html = (f'<div class="dt-wrap"{style}><table class="dt"><thead><tr>{head}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>')
    render_html(html)


def render_topbar():
    st.markdown(
        '<div class="topbar"><div class="brand"><div class="logo">S</div><div>'
        '<div class="brand-name">Shopper Intent</div>'
        '<div class="brand-sub">Model evaluation workspace for online purchase prediction</div></div></div>'
        '<div class="pills"><span class="pill">5 classifiers</span><span class="pill">6 metrics</span>'
        '<span class="pill">17 input features</span><span class="pill brass">M.Tech AIML, Assignment 2</span></div></div>',
        unsafe_allow_html=True,
    )


# =====================================================================
# 7. Page sections
# =====================================================================

def render_setup():
    """Full-width evaluation controls; no sidebar or narrow left rail."""
    if "upload_version" not in st.session_state:
        st.session_state["upload_version"] = 0

    with st.container(key="card_setup"):
        section("Evaluation workspace", "Choose the evaluation scope and upload a labelled CSV. The five fitted pipelines are loaded from disk; this app never retrains them.")
        left, middle, right = st.columns([1.0, 1.35, 1.9], gap="large", vertical_alignment="top")
        with left:
            mode = st.selectbox(
                "Evaluation scope",
                options=[MODE_ONE, MODE_ALL],
                index=0,
                key="evaluation_mode_main",
                help="Choose one classifier for a focused review, or compare all five saved classifiers.",
            )
        with middle:
            model = st.selectbox(
                "Classifier",
                options=list(MODEL_FILES.keys()),
                index=0,
                key="selected_model_main",
                disabled=mode == MODE_ALL,
                help="Select the saved classifier to evaluate when Single model is selected.",
            )
            st.caption(
                MODEL_DESCRIPTIONS[model]
                if mode == MODE_ONE
                else "All five saved classifiers will be evaluated on the same test records."
            )
        with right:
            uploaded = st.file_uploader(
                "Labelled test dataset (CSV)", type=["csv"],
                key=f"test_dataset_upload_{st.session_state['upload_version']}",
                help="Required: 17 input features and Revenue (0/1 or False/True).",
            )
            if uploaded is not None:
                st.markdown(
                    f'<div class="file-chip"><span class="dot"></span><span>{escape(uploaded.name)} · {uploaded.size / 1024:.1f} KB · ready for validation</span></div>',
                    unsafe_allow_html=True,
                )
        action, reset, _ = st.columns([1.25, 1.0, 4.0], gap="small")
        with action:
            label = "Evaluate all five classifiers" if mode == MODE_ALL else f"Evaluate {model}"
            run_clicked = st.button(label, type="primary", width="stretch", key="run_evaluation", disabled=uploaded is None)
        with reset:
            if st.button("New test dataset", width="stretch", key="reset_test_upload"):
                st.session_state["upload_version"] += 1
                st.session_state.pop("evaluation_payload", None)
                st.rerun()
    return mode, model, uploaded, run_clicked


def render_kpis(scores):
    cols = st.columns(6, gap="small")
    for col, metric in zip(cols, METRICS):
        with col:
            kpi_card(metric, scores[metric], METRIC_NOTES[metric])


def render_model_performance(name, y, result):
    scores, predictions, probabilities = result["scores"], result["predictions"], result["probabilities"]

    section(f"{name}", "Scores on the uploaded test dataset.")
    render_kpis(scores)
    st.write("")

    left, right = st.columns(2, gap="medium")
    with left:
        show(confusion_figure(y, predictions), f"cm_{name}")
    with right:
        show(roc_figure(y, probabilities, scores["AUC"]), f"roc_{name}")

    st.write("")
    section("Classification report")
    report = classification_report(
        y, predictions, labels=[0, 1],
        target_names=["No purchase (0)", "Purchase (1)"],
        output_dict=True, zero_division=0,
    )
    
    frame = pd.DataFrame(report).T.reset_index().rename(columns={"index": "Class"})
    # Correct the accuracy row
    accuracy_mask = frame["Class"] == "accuracy"
    # Accuracy represents the overall correctly classified records
    frame.loc[accuracy_mask, "support"] = len(y)

    # Precision and recall are not separately calculated for overall accuracy
    frame.loc[accuracy_mask, ["precision", "recall"]] = np.nan
    # Convert support into integer values
    frame["support"] = frame["support"].astype(int)
    html_table(
        frame,
        renderers={
            "precision": lambda v: "" if pd.isna(v) else f"{v:.4f}",
            "recall": lambda v: "" if pd.isna(v) else f"{v:.4f}",
            "f1-score": lambda v: f"{v:.4f}",
            "support": lambda v: f"{int(v):,}",
        },
    )


def render_comparison(comparison, y, results):
    section("Leaderboard", "All classifiers scored on the same test records, ranked by F1 score.")

    board = comparison.sort_values("F1 Score", ascending=False).reset_index(drop=True)
    board.insert(0, "Rank", range(1, len(board) + 1))
    renderers = {m: bar_cell for m in METRICS}
    renderers["Model"] = lambda v: model_cell(v)
    renderers["Rank"] = lambda v: f"<b>{int(v)}</b>" + ('<span class="crown">TOP</span>' if int(v) == 1 else "")
    html_table(board, renderers=renderers, top_row=0)

    st.write("")
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        show(radar_figure(comparison), "radar_all")
    with c2:
        show(roc_overlay_figure(y, results), "roc_overlay")

    st.write("")
    metric = st.selectbox("Compare by metric", METRICS, index=1, key="comparison_metric")
    show(metric_bar_figure(comparison, metric), "metric_bar")

    st.write("")
    section("Best classifier per metric",
            "The top scorer changes with the metric. Pick the metric that matches your goal, for example recall if missing a buyer is costly.")
    indexed = comparison.set_index("Model")
    best = indexed[METRICS].idxmax()
    best_frame = pd.DataFrame({
        "Metric": METRICS,
        "Top classifier": [best[m] for m in METRICS],
        "Score": [indexed.loc[best[m], m] for m in METRICS],
    })
    html_table(best_frame, renderers={"Top classifier": model_cell, "Score": lambda v: f"<b>{v:.4f}</b>"})


def render_error_analysis(name, y, result):
    tn, fp, fn, tp = confusion_matrix(y, result["predictions"], labels=[0, 1]).ravel()

    section(f"{name}: errors", "Confusion-matrix counts for the uploaded test set.")
    cols = st.columns(4, gap="small")
    items = [
        ("True negatives", tn, "Correctly predicted no purchase"),
        ("False positives", fp, "Predicted a purchase that did not happen"),
        ("False negatives", fn, "Missed an actual purchase"),
        ("True positives", tp, "Correctly identified a purchase"),
    ]
    for col, (label, value, note) in zip(cols, items):
        with col:
            st.metric(label, f"{value:,}", help=note)

    st.write("")
    left, right = st.columns([1.25, 1], gap="medium")
    with left:
        show(confusion_figure(y, result["predictions"]), f"err_cm_{name}")
    with right:
        total = tn + fp + fn + tp
        callout(
            "<b>Reading the matrix</b><br>False positives are sessions labelled as purchases that did not convert. "
            "False negatives are real purchases the model missed. Precision reflects false positives; recall reflects false negatives.",
            "info",
        )
        callout(f"<b>{(fp + fn):,}</b> of {total:,} sessions ({(fp + fn) / total:.1%}) were misclassified.", "best")


def render_predictions(frame, y, result, name):
    output = frame.copy()
    output["Actual Revenue"] = y.to_numpy()
    output["Predicted Revenue"] = result["predictions"]
    output["Purchase Probability"] = result["probabilities"]
    output["Prediction Result"] = np.where(
        output["Actual Revenue"] == output["Predicted Revenue"], "Correct", "Incorrect"
    )

    section(f"{name}: predictions",
            "Original features and the actual target are kept beside each prediction. The download contains every row.")

    c1, _ = st.columns([1, 2.5])
    with c1:
        choice = st.selectbox("Filter records", ["All records", "Incorrect predictions", "Correct predictions"])
    shown = output
    if choice == "Incorrect predictions":
        shown = output[output["Prediction Result"] == "Incorrect"]
    elif choice == "Correct predictions":
        shown = output[output["Prediction Result"] == "Correct"]

    preview = shown.head(100)
    st.caption(f"Previewing {len(preview):,} of {len(shown):,} matching records.")

    # Put the result columns first so they are visible without scrolling.
    lead = ["Prediction Result", "Actual Revenue", "Predicted Revenue", "Purchase Probability"]
    ordered = preview[lead + [c for c in preview.columns if c not in lead]]
    html_table(
        ordered,
        renderers={
            "Prediction Result": lambda v: f'<span class="badge {"ok" if v == "Correct" else "bad"}">{v}</span>',
            "Purchase Probability": lambda v: bar_cell(v),
        },
        float_fmt="{:.4g}",
        max_height=520,
    )
    st.download_button(
        "Download all predictions (CSV)",
        data=output.to_csv(index=False).encode("utf-8"),
        file_name=f"{name.lower().replace(' ', '_')}_test_predictions.csv",
        mime="text/csv",
    )


def sample_csv_bytes():
    sample = {
        "Administrative": [1, 2], "Administrative_Duration": [12.0, 35.5],
        "Informational": [0, 1], "Informational_Duration": [0.0, 18.0],
        "ProductRelated": [5, 12], "ProductRelated_Duration": [120.0, 360.0],
        "BounceRates": [0.02, 0.01], "ExitRates": [0.04, 0.03],
        "PageValues": [0.0, 15.5], "SpecialDay": [0.0, 0.0],
        "Month": ["May", "Nov"], "OperatingSystems": [2, 2], "Browser": [2, 3],
        "Region": [1, 3], "TrafficType": [2, 3],
        "VisitorType": ["Returning_Visitor", "New_Visitor"],
        "Weekend": [False, True], "Revenue": [0, 1],
    }
    return pd.DataFrame(sample).to_csv(index=False).encode("utf-8")


def render_empty_state():
    with st.container(key="card_empty"):
        section("Ready when you are",
                "Compare five saved classifiers on a labelled test set, or inspect one in depth.")
        st.markdown(
            '<div class="steps">'
            '<div class="step"><div class="step-n">1</div><div><b>Choose the scope</b>'
            '<span>Compare all five classifiers, or evaluate a single one.</span></div></div>'
            '<div class="step"><div class="step-n">2</div><div><b>Add the test CSV</b>'
            '<span>It needs the 17 input features and a Revenue column containing both classes (0 and 1).</span></div></div>'
            '<div class="step"><div class="step-n">3</div><div><b>Run the evaluation</b>'
            '<span>You get a leaderboard, ROC curves, error analysis and downloadable predictions.</span></div></div>'
            '</div>',
            unsafe_allow_html=True,
        )


def render_results(mode, selected_model, uploaded, run_clicked):
    if uploaded is None:
        render_empty_state()
        return

    file_bytes = uploaded.getvalue()
    file_hash = hashlib.sha256(file_bytes).hexdigest()
    signature = model_signature()
    run_key = (file_hash, signature, mode, selected_model)
    selected_models = list(MODEL_FILES.keys()) if mode == MODE_ALL else [selected_model]

    missing = [MODEL_FILES[n] for n in selected_models if not (MODEL_DIR / MODEL_FILES[n]).is_file()]
    if missing:
        callout(f"<b>Model files not found</b> in {escape(str(MODEL_DIR))}: {', '.join(missing)}", "warn")
        return

    if run_clicked:
        try:
            with st.spinner("Validating the dataset and running the saved pipelines..."):
                frame, X, y, results, comparison = evaluate_all_models(file_bytes, selected_models)
            st.session_state["evaluation_payload"] = {
                "run_key": run_key, "frame": frame, "X": X, "y": y,
                "results": results, "comparison": comparison,
            }
        except Exception as exc:
            st.session_state.pop("evaluation_payload", None)
            callout(f"<b>Evaluation could not be completed.</b> {escape(str(exc))}", "warn")
            return

    payload = st.session_state.get("evaluation_payload")
    if payload is None or payload.get("run_key") != run_key:
        with st.container(key="card_pending"):
            section("Dataset loaded",
                    "Press the run button in the setup panel to calculate metrics for this file and scope.")
        return

    frame, y = payload["frame"], payload["y"]
    results, comparison = payload["results"], payload["comparison"]

    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    s1, s2, s3, s4 = st.columns(4, gap="small")
    s1.metric("Test records", f"{len(frame):,}")
    s2.metric("Input features", payload["X"].shape[1])
    s3.metric("No purchase", f"{n_neg:,}")
    s4.metric("Purchase", f"{n_pos:,}", help=f"{n_pos / len(y):.1%} of sessions")
    st.write("")

    if mode == MODE_ALL:
        best = comparison.sort_values("F1 Score", ascending=False).iloc[0]
        callout(
            f"<b>Top F1 score:</b> {escape(best['Model'])} at {best['F1 Score']:.4f}. "
            "Weigh the trade-offs between metrics before choosing a final model.",
            "best",
        )
        t_board, t_perf, t_err, t_pred = st.tabs(
            ["Leaderboard", "Model performance", "Error analysis", "Predictions"]
        )
        with t_board:
            st.write("")
            render_comparison(comparison, y, results)

        def pick_model(key):
            c, _ = st.columns([1, 2])
            with c:
                return st.selectbox("Model to inspect", list(results.keys()), key=key)

        with t_perf:
            st.write("")
            name = pick_model("inspect_perf")
            render_model_performance(name, y, results[name])
        with t_err:
            st.write("")
            name = pick_model("inspect_err")
            render_error_analysis(name, y, results[name])
        with t_pred:
            st.write("")
            name = pick_model("inspect_pred")
            render_predictions(frame, y, results[name], name)
    else:
        result = results[selected_model]
        t_perf, t_err, t_pred = st.tabs(["Model performance", "Error analysis", "Predictions"])
        with t_perf:
            st.write("")
            render_model_performance(selected_model, y, result)
        with t_err:
            st.write("")
            render_error_analysis(selected_model, y, result)
        with t_pred:
            st.write("")
            render_predictions(frame, y, result, selected_model)


def render_about():
    section("About this application")
    st.write(
        "This workspace evaluates pre-trained classification pipelines for the UCI Online "
        "Shoppers Purchasing Intention dataset. Uploading a test CSV only runs inference and "
        "computes metrics; no model is trained or modified."
    )

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        with st.container(key="card_objective"):
            st.markdown("#### Project objective")
            st.write(
                "Predict whether an online shopping session results in a purchase "
                "(`Revenue = 1`) or not (`Revenue = 0`) from 17 session-level features."
            )
    with c2:
        with st.container(key="card_prep"):
            st.markdown("#### How the models were prepared")
            st.write(
                "Stratified 80/20 split: 9,864 training and 2,466 test records. Numeric features "
                "are imputed and scaled; categorical features are imputed and one-hot encoded inside "
                "each saved scikit-learn pipeline."
            )

    callout("Keep new test records separate from the original training data to avoid evaluation leakage.", "info")

    st.markdown("#### Available classifiers")
    for model_name, description in MODEL_DESCRIPTIONS.items():
        with st.expander(model_name):
            st.write(description)
            st.write(MODEL_USAGE[model_name])

    st.markdown("#### Input feature reference")
    html_table(
        pd.DataFrame([
            {"Feature": n, "Meaning": FEATURE_DESCRIPTIONS[n],
             "Type": "Categorical / boolean" if n in CATEGORICAL_FEATURES else "Numeric"}
            for n in NUMERIC_FEATURES + CATEGORICAL_FEATURES
        ]),
        wrap_cols=("Meaning",),
    )

    st.markdown("#### Evaluation metrics")
    metric_help = {
        "Accuracy": "Share of all records classified correctly.",
        "AUC": "How well the model ranks purchase sessions above non-purchase sessions.",
        "Precision": "Among sessions predicted as purchases, the share that actually purchased.",
        "Recall": "Among actual purchase sessions, the share detected by the model.",
        "F1 Score": "Harmonic mean of precision and recall.",
        "MCC": "Balanced correlation using all four confusion-matrix outcomes: 1 is perfect, 0 is chance level, -1 is total disagreement.",
    }
    html_table(
        pd.DataFrame([{"Metric": k, "Interpretation": v} for k, v in metric_help.items()]),
        wrap_cols=("Interpretation",),
    )

    st.markdown("#### Sample test CSV")
    st.write("A two-row example of the required 18-column layout. The values are illustrative only.")
    st.download_button(
        "Download sample CSV", data=sample_csv_bytes(),
        file_name="online_shoppers_sample_test.csv", mime="text/csv",
        key="download_sample_csv",
    )


# =====================================================================
# 8. Main
# =====================================================================

def main():
    render_topbar()

    tab_dash, tab_guide = st.tabs(["Dashboard", "Guide"])

    with tab_dash:
        mode, model, uploaded, run_clicked = render_setup()
        st.write("")
        render_results(mode, model, uploaded, run_clicked)

    with tab_guide:
        st.write("")
        render_about()

    st.markdown(
        '<div class="app-footer"><span>UCI Online Shoppers Purchasing Intention, Dataset 468</span>'
        "<span>Saved classifiers, test-only evaluation, no retraining</span></div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()