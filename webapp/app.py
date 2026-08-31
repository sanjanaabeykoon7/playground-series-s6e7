"""
app.py — Flask web application for the Predicting Student Health Risk model.

Function flow when the app runs:
1. On startup: load_artifacts() loads the trained RandomForestClassifier,
   the LabelEncoders for categorical columns, the target encoder, the
   imputation values, and the feature column order (all saved by
   train_model.py using joblib).
2. GET  "/"          -> index()   renders the HTML form (templates/index.html).
3. POST "/predict"   -> predict() reads the submitted form values, builds a
   single-row DataFrame in the exact column order the model expects,
   applies the same imputation + encoding pipeline used during training,
   calls model.predict() and model.predict_proba(), and returns the
   predicted health_condition class (plus class probabilities) to the page.
"""

from flask import Flask, request, render_template
import pandas as pd
import joblib
import os

app = Flask(__name__)

# ---------------------------------------------------------------------------
# 1. Load trained artifacts once at startup (not per-request, for speed)
# ---------------------------------------------------------------------------
ARTIFACT_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(ARTIFACT_DIR, "rf_model.joblib"))
cat_encoders = joblib.load(os.path.join(ARTIFACT_DIR, "encoders.joblib"))
target_encoder = joblib.load(os.path.join(ARTIFACT_DIR, "target_encoder.joblib"))
impute_values = joblib.load(os.path.join(ARTIFACT_DIR, "impute_values.joblib"))
feature_columns = joblib.load(os.path.join(ARTIFACT_DIR, "feature_columns.joblib"))

NUMERIC_COLS = ["sleep_duration", "heart_rate", "bmi", "calorie_expenditure",
                 "step_count", "exercise_duration", "water_intake"]
CATEGORICAL_COLS = ["diet_type", "stress_level", "sleep_quality",
                     "physical_activity_level", "smoking_alcohol", "gender"]


@app.route("/")
def index():
    """Render the input form."""
    return render_template("index.html", categories={
        col: list(enc.classes_) for col, enc in cat_encoders.items()
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    Read form input -> preprocess exactly like training -> predict -> respond.
    """
    form = request.form

    # 1. Build a single-row record from the submitted form
    record = {}
    for col in NUMERIC_COLS:
        val = form.get(col, "").strip()
        record[col] = float(val) if val else impute_values[col]

    for col in CATEGORICAL_COLS:
        val = form.get(col, "").strip()
        record[col] = val if val else impute_values[col]

    row = pd.DataFrame([record])

    # 2. Apply the SAME label encoders fitted during training
    for col in CATEGORICAL_COLS:
        known_classes = set(cat_encoders[col].classes_)
        if row.at[0, col] not in known_classes:
            row.at[0, col] = impute_values[col]  # fallback to training mode
        row[col] = cat_encoders[col].transform(row[col])

    # 3. Ensure column order matches what the model was trained on
    row = row[feature_columns]

    # 4. Predict
    pred_encoded = model.predict(row)[0]
    pred_label = target_encoder.inverse_transform([pred_encoded])[0]
    probabilities = dict(zip(target_encoder.classes_, model.predict_proba(row)[0]))

    return render_template(
        "index.html",
        categories={col: list(enc.classes_) for col, enc in cat_encoders.items()},
        prediction=pred_label,
        probabilities={k: round(v * 100, 2) for k, v in probabilities.items()},
        submitted=form,
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)