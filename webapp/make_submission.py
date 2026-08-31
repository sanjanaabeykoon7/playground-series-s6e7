"""
make_submission.py — generates submission.csv for Kaggle from test.csv,
using the exact same preprocessing pipeline as training.
Run this AFTER train_model.py.
"""

import pandas as pd
import joblib

model = joblib.load("rf_model.joblib")
encoders = joblib.load("encoders.joblib")
target_encoder = joblib.load("target_encoder.joblib")
impute_values = joblib.load("impute_values.joblib")
feature_columns = joblib.load("feature_columns.joblib")

test = pd.read_csv("test.csv")
test_ids = test["id"]
test = test.drop("id", axis=1)

num_cols = test.select_dtypes(include="float64").columns
cat_cols = test.select_dtypes(exclude="float64").columns

for col in num_cols:
    test[col] = test[col].fillna(impute_values[col])

for col in cat_cols:
    test[col] = test[col].fillna(impute_values[col])
    test[col] = test[col].apply(lambda v: v if v in set(encoders[col].classes_) else impute_values[col])
    test[col] = encoders[col].transform(test[col])

test = test[feature_columns]

pred_encoded = model.predict(test)
pred_labels = target_encoder.inverse_transform(pred_encoded)

submission = pd.DataFrame({"id": test_ids, "health_condition": pred_labels})
submission.to_csv("submission.csv", index=False)
print("submission.csv saved with", len(submission), "rows")
print(submission.head())