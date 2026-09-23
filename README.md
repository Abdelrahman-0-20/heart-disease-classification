# Heart Disease Classification - UCI Cleveland Dataset

End-to-end machine learning project with a Streamlit user interface:
data acquisition, EDA and cleaning, preprocessing, three-model
comparison with hyperparameter tuning, evaluation, and an interactive
patient-level prediction form.

## Project structure

```
heart-diseases/
|-- app.py                     # Complete Streamlit application (all logic)
|-- requirements.txt           # Runtime dependencies
|-- .streamlit/config.toml     # Light theme configuration (accent color)
|-- data/
|   `-- processed.cleveland.data   # Downloaded automatically if missing
|-- models/
|   `-- heart_disease_artifact.joblib  # Saved models, metrics and metadata
`-- tests/
    `-- smoke_test.py          # End-to-end verification script
```

## How to run locally

1. Create and activate a virtual environment (recommended):

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   ```

2. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Start the app:

   ```bash
   streamlit run app.py
   ```

   The browser opens at http://localhost:8501. On the first run the app
   downloads the UCI Cleveland dataset (if `data/processed.cleveland.data`
   is missing) and trains the models; the trained artifact is saved to
   `models/` so later runs load the saved model instead of retraining.

## What the app contains

- **Overview** - data preview (first 5 rows), summary statistics, missing
  value report, column types (numerical vs categorical) and the
  preprocessing/split summary (StandardScaler, 80/20 split, random_state=42).
- **EDA** - target distribution, correlation heatmap and feature
  distributions by target class (matplotlib/seaborn).
- **KPIs & Insights** - disease prevalence, average age per class, maximum
  heart rate, cholesterol statistics, plus data-driven insights with
  suggested questions for further analysis.
- **Model Performance** - Accuracy, Precision, Recall, F1 and ROC-AUC for
  Logistic Regression, Random Forest (tuned with GridSearchCV over
  `n_estimators` and `max_depth`) and SVM; confusion matrix and
  classification report for the best model.
- **Prediction** - enter the 13 patient parameters in the sidebar and click
  `Run Prediction` to get the diagnosis label and the disease probability
  from the best model.

## Tests

Headless verification of the data pipeline, training, prediction function,
charts and the full Streamlit script:

```bash
python tests/smoke_test.py
```

## Data source

UCI Machine Learning Repository - Heart Disease Data Set (Cleveland),
processed file with 303 records and 14 columns. The app tries the canonical
UCI URL first, then the UCI zip archive and finally a GitHub mirror, and
validates every download before using it.