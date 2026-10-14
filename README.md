# Student Health Risk Predictor

A machine learning project for the **Kaggle Playground Series S6E7**. It predicts a student's health condition (`fit`, `at-risk` or `unhealthy`) from lifestyle and vital data. A Random Forest classifier does the prediction, and a Flask web app lets you try it in the browser.

## Features

- Exploratory data analysis of the training data (class balance, missing values, correlations, feature vs target plots)
- Random Forest classifier trained on about 690,000 student records
- Class weighting to handle a strongly imbalanced target
- Flask web app with a form for all 13 input features
- The web app shows the predicted class and the probability of each class
- Script to generate a Kaggle `submission.csv` from `test.csv`

## Dataset

The data comes from the Kaggle Playground Series S6E7 competition. It is not included in this repository. Download `train.csv` and `test.csv` from Kaggle.

| Item           | Value                         |
| -------------- | ----------------------------- |
| Training rows  | 690,088                       |
| Input features | 13 (7 numeric, 6 categorical) |
| Target         | `health_condition`            |

**Numeric features:** `sleep_duration`, `heart_rate`, `bmi`, `calorie_expenditure`, `step_count`, `exercise_duration`, `water_intake`

**Categorical features:** `diet_type`, `stress_level`, `sleep_quality`, `physical_activity_level`, `smoking_alcohol`, `gender`

**Class balance:**

| Class     | Share  |
| --------- | ------ |
| at-risk   | 85.87% |
| unhealthy | 8.36%  |
| fit       | 5.77%  |

Every feature has missing values, so the data needs imputation before training.

## Method

1. **Missing values.** Numeric columns are filled with the training mean. Categorical columns are filled with the training mode.
2. **Encoding.** Categorical columns and the target are converted with `LabelEncoder`.
3. **Split.** 80% train and 20% test, stratified by class, with `random_state=42`.
4. **Model.** `RandomForestClassifier` with `n_estimators=200`, `max_depth=12`, `class_weight="balanced"`.
5. **Saved artifacts.** The model, encoders, target encoder, imputation values and feature order are saved with `joblib`. The web app loads them at startup, so it applies the same preprocessing as training.

## Results

Evaluated on the 20% hold-out set (138,018 rows):

| Metric            | Value |
| ----------------- | ----- |
| Accuracy          | 0.874 |
| Balanced accuracy | 0.904 |

| Class     | Precision | Recall | F1-score |
| --------- | --------- | ------ | -------- |
| at-risk   | 0.99      | 0.87   | 0.92     |
| fit       | 0.45      | 0.93   | 0.61     |
| unhealthy | 0.60      | 0.92   | 0.73     |

Class weighting gives high recall on the two minority classes (`fit` and `unhealthy`). The cost is lower precision for those classes, because some `at-risk` students are predicted as `fit` or `unhealthy`.

**Most important features:**

| Feature                   | Importance |
| ------------------------- | ---------- |
| `sleep_duration`          | 0.379      |
| `stress_level`            | 0.311      |
| `physical_activity_level` | 0.159      |

All other features are below 0.05 each.

## Project structure

```
playground-series-s6e7/
├── health_risk_eda.ipynb              # Exploratory data analysis
├── health_risk_RF_classifier.ipynb    # Training, evaluation, saves the artifacts
└── webapp/
    ├── app.py                         # Flask app (form and prediction)
    ├── make_submission.py             # Creates submission.csv for Kaggle
    ├── templates/
    │   └── index.html                 # Input form and result page
    └── *.joblib                       # Saved model and preprocessing artifacts
```

## Getting started

### 1. Install dependencies

```bash
pip install flask pandas scikit-learn joblib matplotlib seaborn jupyter
```

### 2. Get the data

Download `train.csv` and `test.csv` from the Kaggle competition page.

### 3. Train the model

1. Put `train.csv` next to `health_risk_RF_classifier.ipynb`.
2. Run all cells in the notebook.
3. The notebook saves these files: `rf_model.joblib`, `encoders.joblib`, `target_encoder.joblib`, `impute_values.joblib`, `feature_columns.joblib`.
4. Copy the five `.joblib` files into the `webapp/` folder.

The EDA notebook reads `train.csv` from a local path. Change the path in its first code cell to match your machine.

### 4. Run the web app

```bash
cd webapp
python app.py
```

Open <http://127.0.0.1:5000> in your browser, fill in the form and click **Predict Health Condition**.

### 5. Create a Kaggle submission (optional)

Put `test.csv` in the `webapp/` folder, then run:

```bash
cd webapp
python make_submission.py
```

This writes `submission.csv` with the columns `id` and `health_condition`.

## Tech stack

- Python
- scikit-learn (Random Forest, label encoding, metrics)
- pandas, matplotlib, seaborn
- Flask (web app)
- joblib (saving and loading the model)

## Notes

- The saved `.joblib` files should be loaded with the same scikit-learn version that created them.
- The model is a coursework and competition project. It is not a medical tool and should not be used for health decisions.
