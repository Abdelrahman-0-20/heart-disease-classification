

# Imports
 
import datetime
import hashlib
import io
import os
import urllib.request
import zipfile

import joblib
import matplotlib

matplotlib.use("Agg")  # headless-safe backend for server-side rendering
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# 
# Configuration and constants
# 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_PATH = os.path.join(DATA_DIR, "processed.cleveland.data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
ARTIFACT_PATH = os.path.join(MODEL_DIR, "heart_disease_artifact.joblib")
ARTIFACT_VERSION = 1

# Reliable public sources for the same dataset (tried in order).
RAW_DATA_URLS = [
    # Canonical UCI repository file ('?' marks missing values).
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data",
    # Modern UCI distribution: a zip archive holding the same file.
    "https://archive.ics.uci.edu/static/public/45/heart+disease.zip",
    # GitHub mirror (header row, target already binary 0/1,
    # missing 'ca' encoded as 4 and missing 'thal' encoded as 0).
    "https://raw.githubusercontent.com/sharmaroshan/Heart-UCI-Dataset/"
    "master/heart.csv",
]

# Column names of the processed Cleveland dataset (14 columns).
ALL_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
    "exang", "oldpeak", "slope", "ca", "thal", "num",
]
FEATURES = ALL_COLUMNS[:-1]          # 13 patient parameters
TARGET = "target"                    # binary: 0 = no disease, 1 = disease
NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

PREDICTION_THRESHOLD = 0.5           # probability threshold for the positive class

# Human-readable labels for the categorical codes used in the sidebar.
CP_LABELS = {
    1: "1 - Typical angina",
    2: "2 - Atypical angina",
    3: "3 - Non-anginal pain",
    4: "4 - Asymptomatic",
}
RESTECG_LABELS = {
    0: "0 - Normal",
    1: "1 - ST-T abnormality",
    2: "2 - Left ventricular hypertrophy",
}
SLOPE_LABELS = {1: "1 - Upsloping", 2: "2 - Flat", 3: "3 - Downsloping"}
THAL_LABELS = {3: "3 - Normal", 6: "6 - Fixed defect", 7: "7 - Reversable defect"}
FBS_LABELS = {1: "1 - True (above 120 mg/dl)", 0: "0 - False (120 mg/dl or below)"}
EXANG_LABELS = {1: "1 - Yes", 0: "0 - No"}

# Design tokens: light theme, neutral colors, one accent color.
ACCENT_COLOR = "#1F6FEB"             # accent for buttons, bars, highlights
DARK_COLOR = "#2C3E50"               # headers and titles
NEUTRAL_GRAY = "#95A5A6"             # "No Disease" class color
TEXT_COLOR = "#212529"               # body text
CLASS_COLORS = {0: NEUTRAL_GRAY, 1: ACCENT_COLOR}

# Module-level handle on the best model, set after training/loading.
BEST_MODEL_PIPELINE = None
BEST_MODEL_NAME = None


# 
# 1. Data acquisition
# 
def _looks_like_cleveland_file(path):
    """Return True if the file on disk looks like the Cleveland dataset."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            first = handle.readline().strip()
    except OSError:
        return False
    if first.startswith("age"):                       # header-style mirror file
        return first.count(",") >= 13
    parts = first.split(",")                          # raw UCI file: no header
    return len(parts) >= 14 and parts[0].replace(".", "").isdigit()


def _download_to(url, dest_path):
    """Download one URL to dest_path, unpacking the UCI zip if needed."""
    request = urllib.request.Request(
        url, headers={"User-Agent": "heart-disease-classifier/1.0"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = response.read()

    if zipfile.is_zipfile(io.BytesIO(payload)):
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            members = [
                name for name in archive.namelist()
                if name.endswith("processed.cleveland.data")
            ]
            if not members:
                raise ValueError("Zip archive does not contain the dataset file.")
            payload = archive.read(members[0])

    tmp_path = dest_path + ".tmp"
    with open(tmp_path, "wb") as handle:
        handle.write(payload)

    if not _looks_like_cleveland_file(tmp_path):
        os.remove(tmp_path)
        raise ValueError("Downloaded content is not the Cleveland dataset.")

    os.replace(tmp_path, dest_path)


def ensure_dataset():
    """Return the local dataset path, downloading the file if missing."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(RAW_DATA_PATH) and _looks_like_cleveland_file(RAW_DATA_PATH):
        return RAW_DATA_PATH
    last_error = None
    for url in RAW_DATA_URLS:
        try:
            _download_to(url, RAW_DATA_PATH)
            return RAW_DATA_PATH
        except Exception as exc:                      # try the next mirror
            last_error = exc
    raise RuntimeError(
        "Could not download the dataset automatically "
        f"(last error: {last_error}). Please place 'processed.cleveland.data' "
        f"manually at: {RAW_DATA_PATH}"
    )


# 
# 2. Loading, cleaning and imputation
# 
def load_and_prepare_impl():
    """Load the raw data, binarize the target and impute missing values.

    Returns:
        df_missing: DataFrame before imputation (keeps NaNs, for the report).
        df:         DataFrame after imputation (used for EDA and training).
    """
    path = ensure_dataset()
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        first_line = handle.readline().strip()

    if first_line.startswith("age"):
        # Mirror CSV: header row, target already binary, missing values
        # encoded as ca = 4 and thal = 0.
        df = pd.read_csv(path)
        df = df.rename(columns={"target": TARGET})
        df.loc[df["ca"] == 4, "ca"] = np.nan
        df.loc[df["thal"] == 0, "thal"] = np.nan
    else:
        # Raw UCI file: no header, '?' marks missing values, target
        # 'num' is 0-4 and gets binarized (> 0 means disease).
        df = pd.read_csv(path, header=None, names=ALL_COLUMNS, na_values="?")
        df[TARGET] = (df["num"] > 0).astype(int)
        df = df.drop(columns=["num"])

    # Coerce every column to numeric so that imputation and models behave.
    for column in FEATURES:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df[TARGET] = df[TARGET].astype(int)

    df_missing = df.copy()
    df = impute_missing(df)
    return df_missing, df


def impute_missing(df):
    """Impute numerical columns with the median, categorical with the mode."""
    out = df.copy()
    for column in NUMERIC_FEATURES:
        out[column] = out[column].fillna(out[column].median())
    for column in CATEGORICAL_FEATURES:
        mode_value = out[column].mode(dropna=True)
        fill = mode_value.iloc[0] if len(mode_value) else 0
        out[column] = out[column].fillna(fill)
    return out


# 
# 3. Preprocessing and model building
# 
def build_preprocessor():
    """Column transformer: scale numerical features, pass categorical codes."""
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, NUMERIC_FEATURES),
        ("cat", categorical_pipe, CATEGORICAL_FEATURES),
    ])


def data_signature(df):
    """Cheap fingerprint of the prepared data (used to validate the cache)."""
    digest = hashlib.md5(
        pd.util.hash_pandas_object(df, index=True).values.tobytes()
    ).hexdigest()
    return f"{df.shape[0]}x{df.shape[1]}-{digest}"


def _load_saved_artifact(signature):
    """Load the trained artifact from disk if it matches this dataset."""
    if not os.path.exists(ARTIFACT_PATH):
        return None
    try:
        artifact = joblib.load(ARTIFACT_PATH)
    except Exception:
        return None
    if artifact.get("version") != ARTIFACT_VERSION:
        return None
    if artifact.get("data_signature") != signature:
        return None
    return artifact


def set_best_model(artifact):
    """Point predict_heart_disease at the best model of the artifact.

    Called on every script run: st.cache_resource may return a cached
    artifact without re-executing the training function, and each
    Streamlit rerun resets the module-level handles.
    """
    global BEST_MODEL_PIPELINE, BEST_MODEL_NAME
    BEST_MODEL_PIPELINE = artifact["models"][artifact["best_name"]]
    BEST_MODEL_NAME = artifact["best_name"]


def _compute_row_metrics(name, pipeline, X_test, y_test):
    """Accuracy, Precision, Recall, F1 and ROC-AUC for one fitted model."""
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_prob),
    }


def train_and_evaluate_impl(df):
    """Train all models, evaluate them, pick the best one and cache it.

    Models: Logistic Regression (baseline), Random Forest (tuned with
    GridSearchCV over n_estimators / max_depth) and SVM (RBF kernel).
    The best model is the one with the highest F1-score (ROC-AUC as
    tie-breaker) on the 20% holdout test set.
    """
    signature = data_signature(df)

    # Fast path: reuse the saved model if the data has not changed.
    saved = _load_saved_artifact(signature)
    if saved is not None:
        set_best_model(saved)
        return saved

    # ----- train / test split: 80% train, 20% test, random_state=42 -----
    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ----- baseline models -----
    models = {
        "Logistic Regression": Pipeline([
            ("preprocessor", build_preprocessor()),
            ("model", LogisticRegression(max_iter=1000, random_state=42)),
        ]),
        "SVM": Pipeline([
            ("preprocessor", build_preprocessor()),
            ("model", SVC(kernel="rbf", probability=True, random_state=42)),
        ]),
    }
    for pipeline in models.values():
        pipeline.fit(X_train, y_train)

    # ----- Random Forest tuned with GridSearchCV -----
    rf_grid = GridSearchCV(
        Pipeline([
            ("preprocessor", build_preprocessor()),
            ("model", RandomForestClassifier(random_state=42)),
        ]),
        param_grid={
            "model__n_estimators": [100, 200, 300],
            "model__max_depth": [None, 4, 6, 10],
        },
        cv=5,
        scoring="f1",
        n_jobs=-1,
    )
    rf_grid.fit(X_train, y_train)
    models["Random Forest"] = rf_grid.best_estimator_

    # ----- evaluation on the holdout set -----
    metric_rows = [
        _compute_row_metrics(name, pipeline, X_test, y_test)
        for name, pipeline in models.items()
    ]
    best_row = max(metric_rows, key=lambda row: (row["F1"], row["ROC-AUC"]))
    best_name = best_row["Model"]

    best_pipeline = models[best_name]
    y_pred_best = best_pipeline.predict(X_test)
    confusion = confusion_matrix(y_test, y_pred_best)
    report = classification_report(
        y_test, y_pred_best, target_names=["No Disease", "Disease"], digits=3
    )

    artifact = {
        "version": ARTIFACT_VERSION,
        "data_signature": signature,
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "models": models,
        "metrics": metric_rows,
        "best_name": best_name,
        "rf_best_params": rf_grid.best_params_,
        "rf_best_cv_score": float(rf_grid.best_score_),
        "confusion_matrix": confusion,
        "classification_report": report,
        "split_info": {
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
        },
        "feature_names": FEATURES,
    }

    # Persist so that the next app start loads the saved model instead
    # of retraining.
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(artifact, ARTIFACT_PATH)

    set_best_model(artifact)
    return artifact


@st.cache_data(show_spinner="Loading dataset...")
def load_and_prepare():
    """Cached wrapper around load_and_prepare_impl (UI entry point)."""
    return load_and_prepare_impl()


@st.cache_resource(show_spinner="Training models (first run only)...")
def get_artifact(df):
    """Cached wrapper around train_and_evaluate_impl (UI entry point)."""
    return train_and_evaluate_impl(df)


# 
# 5. Prediction interface
# 
def predict_heart_disease(patient_data):
    """Predict heart disease for a single patient using the best model.

    Args:
        patient_data: dict with the 13 patient parameters, e.g.
            {"age": 63, "sex": 1, "cp": 4, "trestbps": 145, "chol": 233,
             "fbs": 1, "restecg": 2, "thalach": 150, "exang": 0,
             "oldpeak": 2.3, "slope": 3, "ca": 0, "thal": 6}

    Returns:
        dict with keys: prediction (0/1), label, probability (float
        0-1, probability of disease), probability_pct, model.
    """
    if BEST_MODEL_PIPELINE is None:
        raise RuntimeError(
            "Model is not trained yet. Run the training pipeline first."
        )
    features = pd.DataFrame([patient_data], columns=FEATURES)
    probability = float(BEST_MODEL_PIPELINE.predict_proba(features)[0, 1])
    prediction = int(probability >= PREDICTION_THRESHOLD)
    return {
        "prediction": prediction,
        "label": "Heart Disease Detected" if prediction else "No Heart Disease",
        "probability": probability,
        "probability_pct": probability * 100.0,
        "model": BEST_MODEL_NAME,
    }


# 
# Chart helpers (matplotlib / seaborn, minimal styling)
# 
def setup_plot_style():
    """Global matplotlib/seaborn style: light, minimal, legible."""
    sns.set_style("white")
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#DEE2E6",
        "axes.labelcolor": TEXT_COLOR,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlecolor": DARK_COLOR,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 110,
    })


def clean_axes(ax):
    """Minimal axes: no top/right spines, light horizontal grid."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color="#E9ECEF", linewidth=0.8)
    ax.set_axisbelow(True)


def plot_target_distribution(df):
    """Bar chart of the target class balance."""
    counts = df[TARGET].value_counts().sort_index()
    total = int(counts.sum())
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    bars = ax.bar(
        ["No Disease (0)", "Disease (1)"],
        counts.values,
        color=[CLASS_COLORS[0], CLASS_COLORS[1]],
        width=0.55,
    )
    for bar, value in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + total * 0.02,
            f"{value} ({value / total * 100:.1f}%)",
            ha="center", fontsize=10, color=TEXT_COLOR,
        )
    ax.set_title("Target distribution")
    ax.set_ylabel("Patients")
    ax.set_ylim(0, counts.max() * 1.18)
    clean_axes(ax)
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(df):
    """Correlation heatmap of all features plus the target."""
    correlation = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(8.8, 7.2))
    sns.heatmap(
        correlation, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
        vmin=-1, vmax=1, square=True, linewidths=0.5, linecolor="white",
        cbar_kws={"shrink": 0.8}, annot_kws={"size": 7.5}, ax=ax,
    )
    ax.set_title("Correlation heatmap")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    fig.tight_layout()
    return fig


def plot_feature_distributions(df):
    """Grid of numerical feature distributions, split by target class."""
    hue_values = df[TARGET].map({0: "No Disease", 1: "Disease"}).rename("Diagnosis")
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.0))
    axes = axes.ravel()
    for index, (ax, column) in enumerate(zip(axes, NUMERIC_FEATURES)):
        sns.histplot(
            data=df, x=column, hue=hue_values,
            hue_order=["No Disease", "Disease"],
            palette={"No Disease": NEUTRAL_GRAY, "Disease": ACCENT_COLOR},
            kde=True, element="step", stat="density", common_norm=False,
            fill=True, alpha=0.35, ax=ax, legend=(index == 0),
        )
        ax.set_title(FEATURE_LABELS[column])
        ax.set_xlabel("")
        ax.set_ylabel("Density")
        clean_axes(ax)
    axes[5].axis("off")   # 5 features in a 2x3 grid: hide the empty slot
    fig.tight_layout()
    return fig


def plot_confusion_matrix(cm):
    """Confusion matrix heatmap for the best model."""
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False,
        annot_kws={"size": 13}, linewidths=0.5, linecolor="white",
        xticklabels=["No Disease", "Disease"],
        yticklabels=["No Disease", "Disease"], ax=ax,
    )
    ax.set_title("Confusion matrix - best model")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    fig.tight_layout()
    return fig


FEATURE_LABELS = {
    "age": "Age (years)",
    "trestbps": "Resting blood pressure (mm Hg)",
    "chol": "Cholesterol (mg/dl)",
    "thalach": "Max heart rate (bpm)",
    "oldpeak": "ST depression (oldpeak)",
}


# 
# KPIs and insights
# 
def compute_kpis(df):
    """Key performance indicators displayed with st.metric."""
    disease_mask = df[TARGET] == 1
    return {
        "n_total": int(len(df)),
        "n_disease": int(disease_mask.sum()),
        "prevalence_pct": float(df[TARGET].mean() * 100.0),
        "age_disease": float(df.loc[disease_mask, "age"].mean()),
        "age_no_disease": float(df.loc[~disease_mask, "age"].mean()),
        "max_thalach": float(df["thalach"].max()),
        "mean_chol": float(df["chol"].mean()),
        "max_chol": float(df["chol"].max()),
    }


def build_insights(df):
    """Data-driven insights, each with a suggested follow-up question."""
    n_total = len(df)
    n_disease = int((df[TARGET] == 1).sum())
    prevalence = df[TARGET].mean() * 100.0

    def disease_rate(mask):
        return df.loc[mask, TARGET].mean() * 100.0

    asymptomatic = df["cp"] == 4
    rate_asymptomatic = disease_rate(asymptomatic)
    rate_other_cp = disease_rate(~asymptomatic)

    rate_exang = disease_rate(df["exang"] == 1)
    rate_no_exang = disease_rate(df["exang"] == 0)

    thalach_disease = df.loc[df[TARGET] == 1, "thalach"].mean()
    thalach_healthy = df.loc[df[TARGET] == 0, "thalach"].mean()

    age_disease = df.loc[df[TARGET] == 1, "age"].mean()
    age_healthy = df.loc[df[TARGET] == 0, "age"].mean()
    chol_disease = df.loc[df[TARGET] == 1, "chol"].median()
    chol_healthy = df.loc[df[TARGET] == 0, "chol"].median()

    return [
        (
            "Disease prevalence",
            f"{prevalence:.1f}% of the {n_total} patients ({n_disease}) are "
            "diagnosed with heart disease, so the classes are moderately "
            "imbalanced.",
            "Would class weighting or a lower decision threshold improve "
            "sensitivity for the disease class?",
        ),
        (
            "Chest pain type is a strong signal",
            f"Among patients with asymptomatic chest pain (cp = 4) "
            f"{rate_asymptomatic:.0f}% have heart disease, versus "
            f"{rate_other_cp:.0f}% of patients with the other chest pain types.",
            "Does chest pain type strongly predict disease? Quantify it with "
            "permutation importance or mutual information.",
        ),
        (
            "Exercise-induced angina raises risk",
            f"Patients with exercise-induced angina (exang = 1) show a disease "
            f"rate of {rate_exang:.0f}% versus {rate_no_exang:.0f}% for "
            "patients without it.",
            "How much independent signal does exercise-induced angina add once "
            "chest pain type is already known?",
        ),
        (
            "Lower maximum heart rate is associated with disease",
            f"Patients with heart disease reach an average maximum heart rate "
            f"of {thalach_disease:.0f} bpm versus {thalach_healthy:.0f} bpm "
            "for patients without disease.",
            "Is maximum heart rate a direct protective factor or a proxy for "
            "age and fitness?",
        ),
        (
            "Age separates the groups; cholesterol does not",
            f"Average age is {age_disease:.1f} years in the disease group "
            f"versus {age_healthy:.1f} years, while median cholesterol is "
            f"similar ({chol_disease:.0f} vs {chol_healthy:.0f} mg/dl).",
            "Do cholesterol and the number of major vessels (ca) add "
            "predictive power beyond age?",
        ),
    ]


# 
# 6. Streamlit UI
# 
def apply_custom_css():
    """Inject the light theme colors and accent styling."""
    st.markdown(
        """
        <style>
          .stApp { background-color: #FFFFFF; color: #212529; }
          h1, h2, h3 { color: #2C3E50; }
          section[data-testid="stSidebar"] {
              background-color: #F8F9FA;
              border-right: 1px solid #E9ECEF;
          }
          button[kind="primary"] {
              background-color: #1F6FEB;
              border-color: #1F6FEB;
              color: #FFFFFF;
          }
          button[kind="primary"]:hover {
              background-color: #1858BC;
              border-color: #1858BC;
          }
          div[data-testid="stMetricValue"] { color: #2C3E50; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar():
    """Patient input form in the sidebar. Returns (patient_dict, clicked)."""
    with st.sidebar:
        st.header("Patient Input")
        st.caption("Enter the clinical parameters, then run the prediction.")

        with st.expander("Demographics", expanded=True):
            age = st.number_input(
                "Age (years)", min_value=25, max_value=85, value=54,
                step=1, help="Patient age in years",
            )
            sex = st.selectbox(
                "Sex", options=[1, 0],
                format_func=lambda v: "Male" if v == 1 else "Female", index=0,
            )

        with st.expander("Vitals and Lab Results", expanded=True):
            cp = st.selectbox(
                "Chest pain type (cp)", options=[1, 2, 3, 4],
                format_func=lambda v: CP_LABELS[v], index=3,
            )
            trestbps = st.number_input(
                "Resting blood pressure (mm Hg)", min_value=90, max_value=210,
                value=130, step=1,
            )
            chol = st.number_input(
                "Serum cholesterol (mg/dl)", min_value=100, max_value=600,
                value=240, step=1,
            )
            fbs = st.selectbox(
                "Fasting blood sugar (fbs)", options=[1, 0],
                format_func=lambda v: FBS_LABELS[v], index=1,
            )
            restecg = st.selectbox(
                "Resting ECG (restecg)", options=[0, 1, 2],
                format_func=lambda v: RESTECG_LABELS[v], index=0,
            )

        with st.expander("Exercise Test Results", expanded=True):
            thalach = st.number_input(
                "Max heart rate achieved (bpm)", min_value=60, max_value=210,
                value=150, step=1,
            )
            exang = st.selectbox(
                "Exercise induced angina (exang)", options=[1, 0],
                format_func=lambda v: EXANG_LABELS[v], index=1,
            )
            oldpeak = st.number_input(
                "ST depression induced by exercise (oldpeak)", min_value=0.0,
                max_value=8.0, value=1.0, step=0.1, format="%.1f",
            )
            slope = st.selectbox(
                "Slope of peak exercise ST segment (slope)", options=[1, 2, 3],
                format_func=lambda v: SLOPE_LABELS[v], index=0,
            )
            ca = st.selectbox(
                "Major vessels colored by fluoroscopy (ca)", options=[0, 1, 2, 3],
                index=0, help="Number of major vessels (0-3)",
            )
            thal = st.selectbox(
                "Thalassemia (thal)", options=[3, 6, 7],
                format_func=lambda v: THAL_LABELS[v], index=0,
            )

        st.divider()
        clicked = st.button("Run Prediction", type="primary", width="stretch")

    patient = {
        "age": int(age), "sex": int(sex), "cp": int(cp),
        "trestbps": int(trestbps), "chol": int(chol), "fbs": int(fbs),
        "restecg": int(restecg), "thalach": int(thalach),
        "exang": int(exang), "oldpeak": float(oldpeak), "slope": int(slope),
        "ca": int(ca), "thal": int(thal),
    }
    return patient, clicked


def render_overview(df_missing, df, artifact):
    """Overview tab: project summary, data preview, types, preprocessing."""
    st.subheader("About this project")
    st.write(
        "Binary classification of heart disease on the UCI Cleveland dataset "
        "(303 patients, 13 clinical parameters). Three models are compared "
        "and the best one powers the interactive prediction."
    )
    left, right = st.columns(2)
    with left:
        st.markdown("**Goal**")
        st.write(
            "Predict whether a patient has heart disease from routine clinical "
            "measurements and show the probability behind each prediction."
        )
    with right:
        st.markdown("**Workflow**")
        st.write(
            "Download and clean the data, explore it, impute missing values, "
            "scale numerical features, train and tune three models, evaluate "
            "them on a holdout set and serve the best model."
        )

    st.divider()
    st.subheader("First 5 rows")
    st.dataframe(df_missing.head(5), hide_index=True)

    st.divider()
    st.subheader("Summary statistics")
    st.dataframe(df.describe().T)

    st.divider()
    st.subheader("Missing values (before imputation)")
    missing = df_missing[FEATURES].isna().sum()
    missing = missing[missing > 0]
    if len(missing):
        st.dataframe(missing.rename("Missing values").to_frame())
    else:
        st.write("No missing values detected.")
    st.caption(
        "Missing numerical values are imputed with the column median; "
        "missing categorical values are imputed with the mode."
    )

    st.divider()
    st.subheader("Column types")
    col_num, col_cat = st.columns(2)
    with col_num:
        st.markdown("**Numerical features** (scaled with StandardScaler)")
        st.write(", ".join(NUMERIC_FEATURES))
    with col_cat:
        st.markdown("**Categorical features** (integer codes, passed through)")
        st.write(", ".join(CATEGORICAL_FEATURES))

    st.divider()
    st.subheader("Preprocessing and train / test split")
    split_info = artifact["split_info"]
    st.markdown(
        f"- Numerical features standardized with StandardScaler "
        f"(fit on the training set only).\n"
        f"- Split: {split_info['n_train']} training rows and "
        f"{split_info['n_test']} test rows (80% / 20%, random_state=42, "
        "stratified by target).\n"
        f"- Best model: {artifact['best_name']}."
    )


def render_eda(df):
    """EDA tab: the three required visualizations."""
    st.subheader("Target distribution")
    st.caption("Class balance of the binary target (0 = No Disease, 1 = Disease).")
    st.pyplot(plot_target_distribution(df))

    st.divider()
    st.subheader("Correlation heatmap")
    st.caption("Pearson correlation between all features and the target.")
    st.pyplot(plot_correlation_heatmap(df))

    st.divider()
    st.subheader("Feature distributions by target")
    st.caption(
        "Distributions of the numerical features, split by target class."
    )
    st.pyplot(plot_feature_distributions(df))


def render_kpis(df):
    """KPIs & Insights tab: metric tiles plus analysis highlights."""
    st.subheader("Key performance indicators")
    kpis = compute_kpis(df)

    row1 = st.columns(3)
    row1[0].metric(
        "Disease Prevalence", f"{kpis['prevalence_pct']:.1f}%",
        help=f"{kpis['n_disease']} of {kpis['n_total']} patients",
    )
    row1[1].metric(
        "Avg Age - Disease", f"{kpis['age_disease']:.1f} yr",
        help="Mean age of patients with heart disease",
    )
    row1[2].metric(
        "Avg Age - No Disease", f"{kpis['age_no_disease']:.1f} yr",
        help="Mean age of patients without heart disease",
    )

    row2 = st.columns(3)
    row2[0].metric(
        "Max Heart Rate", f"{kpis['max_thalach']:.0f} bpm",
        help="Highest maximum heart rate achieved (thalach) in the dataset",
    )
    row2[1].metric(
        "Mean Cholesterol", f"{kpis['mean_chol']:.0f} mg/dl",
        help="Mean serum cholesterol in the dataset",
    )
    row2[2].metric(
        "Max Cholesterol", f"{kpis['max_chol']:.0f} mg/dl",
        help="Highest serum cholesterol in the dataset",
    )

    st.divider()
    st.subheader("Insights")
    st.caption("Computed from the full dataset, each with a suggested question "
               "for further analysis.")
    for index, (title, text, question) in enumerate(build_insights(df), start=1):
        st.markdown(f"{index}. **{title}**")
        st.write(text)
        st.caption(f"Suggested question: {question}")


def render_model_performance(artifact):
    """Model Performance tab: comparison table, tuning result, best model."""
    st.subheader("Model comparison (20% holdout set)")
    metrics_df = pd.DataFrame(artifact["metrics"]).set_index("Model").round(3)
    st.dataframe(metrics_df)
    st.caption(
        f"Best model: {artifact['best_name']} "
        "(selected by F1-score, ROC-AUC as tie-breaker)."
    )

    st.divider()
    st.subheader("Random Forest - hyperparameter tuning")
    pretty_params = ", ".join(
        f"{name.replace('model__', '')} = {value}"
        for name, value in artifact["rf_best_params"].items()
    )
    st.write(f"GridSearchCV (5-fold, scoring = F1) best parameters: {pretty_params}")
    st.write(f"Best cross-validated F1-score: {artifact['rf_best_cv_score']:.3f}")

    st.divider()
    st.subheader(f"Best model - {artifact['best_name']}")
    matrix_col, report_col = st.columns([2, 3])
    with matrix_col:
        st.pyplot(plot_confusion_matrix(artifact["confusion_matrix"]))
    with report_col:
        st.caption("Classification report (precision, recall, F1 per class).")
        st.text(artifact["classification_report"])


def render_prediction(patient, clicked):
    """Prediction tab: runs predict_heart_disease and displays the result."""
    st.subheader("Predict heart disease for a new patient")
    st.write(
        "Set the patient parameters in the sidebar, then click "
        "'Run Prediction'."
    )

    if clicked:
        try:
            result = predict_heart_disease(patient)
            st.session_state["last_prediction"] = result
            st.session_state["last_patient"] = dict(patient)
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

    result = st.session_state.get("last_prediction")
    if result is None:
        st.info(
            "No prediction yet. Enter patient parameters in the sidebar and "
            "click 'Run Prediction'."
        )
        return

    last_patient = st.session_state.get("last_patient", {})
    if last_patient:
        st.caption("Patient parameters used for this prediction")
        st.dataframe(
            pd.Series(last_patient).rename("Value").to_frame(), hide_index=True
        )

    probability = result["probability"]
    if result["prediction"] == 1:
        st.error(
            f"**Prediction: Heart Disease Detected**  \n"
            f"**Probability: {result['probability_pct']:.1f}%**"
        )
    else:
        st.success(
            f"**Prediction: No Heart Disease**  \n"
            f"**Probability: {result['probability_pct']:.1f}%**"
        )

    st.progress(probability)
    st.caption(
        f"Model: {result['model']} | Probability of heart disease | "
        f"Decision threshold: {PREDICTION_THRESHOLD:.2f}"
    )


# 
# Main entry point
# 
def main():
    setup_plot_style()
    st.set_page_config(
        page_title="Heart Disease Classification",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_custom_css()

    st.title("Heart Disease Classification")
    st.caption(
        "UCI Cleveland dataset | Data exploration, model comparison and "
        "patient-level prediction."
    )

    # ----- data and models (cached: downloads and trains on first run) -----
    df_missing, df = load_and_prepare()
    artifact = get_artifact(df)
    set_best_model(artifact)  # re-bind on every rerun (cache may skip training)

    # ----- sidebar patient form -----
    patient, clicked = render_sidebar()

    # ----- main tabs -----
    tab_overview, tab_eda, tab_kpi, tab_model, tab_predict = st.tabs([
        "Overview", "EDA", "KPIs & Insights", "Model Performance", "Prediction",
    ])
    with tab_overview:
        render_overview(df_missing, df, artifact)
    with tab_eda:
        render_eda(df)
    with tab_kpi:
        render_kpis(df)
    with tab_model:
        render_model_performance(artifact)
    with tab_predict:
        render_prediction(patient, clicked)


if __name__ == "__main__":
    main()
