"""
Smoke test for the Heart Disease Classification project.

Runs outside the browser and verifies, in order:
  1. Data acquisition, cleaning and imputation.
  2. Model training, tuning, evaluation and artifact persistence.
  3. predict_heart_disease() for typical and healthy patients.
  4. All matplotlib/seaborn charts render without errors.
  5. The full Streamlit script executes (via streamlit AppTest),
     including a sidebar button click and the prediction output.

Run with:  .venv/bin/python tests/smoke_test.py
"""
import os
import sys
import tempfile

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp())

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import pandas as pd  # noqa: E402

import app  # noqa: E402

failures = []


def check(label, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}" + (f" - {detail}" if detail else ""))
    if not condition:
        failures.append(label)



# 1. Data pipeline
 
df_missing, df = app.load_and_prepare_impl()
print("data shape:", df.shape)
print("target counts:", df[TARGET := "target"].value_counts().to_dict())
missing_counts = df_missing.isna().sum()
print("missing before imputation:", missing_counts[missing_counts > 0].to_dict())
print("missing after imputation:", int(df.isna().sum().sum()))
check("dataset has 303 rows x 14 columns", df.shape == (303, 14), str(df.shape))
check("no missing values after imputation", int(df.isna().sum().sum()) == 0)
check("target is binary", set(df["target"].unique()) <= {0, 1})

# 
# 2. Training, tuning and evaluation
# 
artifact = app.train_and_evaluate_impl(df)
print(pd.DataFrame(artifact["metrics"]).round(3).to_string(index=False))
print("best model:", artifact["best_name"])
print("RF best params:", artifact["rf_best_params"])
print("RF best CV score:", round(artifact["rf_best_cv_score"], 3))
check("three models trained", len(artifact["metrics"]) == 3)
check("metric rows contain 5 metrics",
      all(len(row) == 6 for row in artifact["metrics"]))  # Model + 5 metrics
check("best model selected", artifact["best_name"] in ("Logistic Regression",
      "Random Forest", "SVM"))
check("grid searched n_estimators", "model__n_estimators" in artifact["rf_best_params"])
check("grid searched max_depth", "model__max_depth" in artifact["rf_best_params"])
check("confusion matrix 2x2", artifact["confusion_matrix"].shape == (2, 2))
check("artifact saved to disk", os.path.exists(app.ARTIFACT_PATH))

# Saved-model fast path: calling again must return the same artifact.
artifact_again = app.train_and_evaluate_impl(df)
check("saved model reloaded from disk",
      artifact_again["created_at"] == artifact["created_at"])
check("best model handle is set", app.BEST_MODEL_PIPELINE is not None)

# 
# 3. Prediction interface
# 
sick_patient = {
    "age": 63, "sex": 1, "cp": 4, "trestbps": 145, "chol": 233, "fbs": 1,
    "restecg": 2, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3,
    "ca": 0, "thal": 6,
}
healthy_patient = {
    "age": 41, "sex": 0, "cp": 2, "trestbps": 120, "chol": 200, "fbs": 0,
    "restecg": 0, "thalach": 175, "exang": 0, "oldpeak": 0.2, "slope": 1,
    "ca": 0, "thal": 3,
}
for label, patient in (("sick-pattern", sick_patient), ("healthy-pattern", healthy_patient)):
    result = app.predict_heart_disease(patient)
    print(f"prediction ({label}):", result)
    check(f"predict_heart_disease returns all keys ({label})",
          {"prediction", "label", "probability", "probability_pct", "model"}
          <= set(result))
    check(f"probability in [0, 1] ({label})",
          0.0 <= result["probability"] <= 1.0)
check("threshold logic consistent",
      (result := app.predict_heart_disease(sick_patient))["prediction"]
      == int(result["probability"] >= app.PREDICTION_THRESHOLD))

# 
# 4. Charts
# 
figures = [
    app.plot_target_distribution(df),
    app.plot_correlation_heatmap(df),
    app.plot_feature_distributions(df),
    app.plot_confusion_matrix(artifact["confusion_matrix"]),
]
chart_dir = tempfile.mkdtemp()
for index, figure in enumerate(figures):
    path = os.path.join(chart_dir, f"fig{index}.png")
    figure.savefig(path, dpi=90)
    size = os.path.getsize(path)
    check(f"chart {index} rendered", size > 5000, f"{size} bytes")

# 
# 5. Full Streamlit script execution (AppTest)
# 
from streamlit.testing.v1 import AppTest  # noqa: E402

at = AppTest.from_file(os.path.join(PROJECT_ROOT, "app.py"), default_timeout=300)
at.run()
check("streamlit script runs without exception", not at.exception, str(at.exception))
tab_labels = [tab.label for tab in at.tabs]
print("tabs:", tab_labels)
check("five tabs present", tab_labels == ["Overview", "EDA", "KPIs & Insights",
      "Model Performance", "Prediction"])
check("sidebar has Run Prediction button",
      len(at.sidebar.button) == 1 and at.sidebar.button[0].label == "Run Prediction")

# Click the prediction button and re-run the script.
at.sidebar.button[0].click()
at.run()
check("no exception after prediction click", not at.exception, str(at.exception))
blocks = [element.value for element in list(at.error) + list(at.success)]
print("prediction blocks:", blocks)
check("prediction result displayed",
      any("Prediction:" in text and "Probability:" in text for text in blocks))


# Summary
 
print()
if failures:
    print(f"SMOKE TEST FAILED: {failures}")
    sys.exit(1)
print("ALL SMOKE TESTS PASSED")
