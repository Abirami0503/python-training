import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import cross_val_score
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


# 1. LOAD DATASET

BASE_DIR = Path(__file__).resolve().parent

data_path = BASE_DIR / "data" / "loan_data.csv"

df = pd.read_csv(data_path, parse_dates=["loan_date"])

print("\n========== DATASET LOADED ==========")
print("Dataset Shape:", df.shape)

X = df.drop(columns=["loan_default"])
y = df["loan_default"]


# 2. HANDLE MISSING VALUES

numeric_features = [
    "age",
    "income",
    "credit_score",
    "loan_amount",
    "interest_rate",
    "loan_term",
    "existing_debt",
    "debt_to_income"
]

categorical_features = [
    "employment_type",
    "education_level"
]

print("\n========== MISSING VALUE HANDLING ==========")

print("\nMissing Values:")
print(X.isnull().sum())

# SimpleImputer for numerical columns

numeric_imputer = SimpleImputer(strategy="median")

X_numeric = numeric_imputer.fit_transform(
    X[numeric_features]
)

print("\nSimpleImputer - Median Strategy")
print("Numerical Data Shape:", X_numeric.shape)
print("Remaining Missing Values:", np.isnan(X_numeric).sum())

# SimpleImputer for categorical columns

categorical_imputer = SimpleImputer(strategy="most_frequent")

X_categorical = categorical_imputer.fit_transform(
    X[categorical_features]
)

print("\nSimpleImputer - Most Frequent Strategy")
print("Categorical Data Shape:", X_categorical.shape)

# KNNImputer for numerical columns

knn_imputer = KNNImputer(n_neighbors=5)

X_knn = knn_imputer.fit_transform(
    X[numeric_features]
)

print("\nKNNImputer Results")
print("Numerical Data Shape:", X_knn.shape)
print("Remaining Missing Values:", np.isnan(X_knn).sum())


# 3. FEATURE ENGINEERING

class LoanFeatureEngineer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        X = X.copy()

        # Date decomposition
        X["loan_year"] = X["loan_date"].dt.year
        X["loan_month"] = X["loan_date"].dt.month
        X["loan_day"] = X["loan_date"].dt.day

        # Interaction feature
        X["income_loan_ratio"] = (
            X["income"] / X["loan_amount"].replace(0, np.nan)
        )

        # Log transformation
        X["loan_amount_log"] = np.log1p(
            X["loan_amount"].clip(lower=0)
        )

        X = X.drop(columns=["loan_date"])

        return X


print("\n========== FEATURE ENGINEERING ==========")

print("New Features Created:")
print([
    "loan_year",
    "loan_month",
    "loan_day",
    "income_loan_ratio",
    "loan_amount_log"
])


# 4. ENCODING AND SCALING

engineered_numeric_features = [
    "age",
    "income",
    "credit_score",
    "loan_amount",
    "interest_rate",
    "loan_term",
    "existing_debt",
    "debt_to_income",
    "loan_year",
    "loan_month",
    "loan_day",
    "income_loan_ratio",
    "loan_amount_log"
]

onehot_features = ["employment_type"]

ordinal_features = ["education_level"]

numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

onehot_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

ordinal_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(
        categories=[["High School", "Graduate", "Postgraduate"]],
        handle_unknown="use_encoded_value",
        unknown_value=-1
    ))
])

preprocessor = ColumnTransformer(transformers=[
    ("numeric", numeric_pipeline, engineered_numeric_features),
    ("onehot", onehot_pipeline, onehot_features),
    ("ordinal", ordinal_pipeline, ordinal_features)
])

print("\n========== ENCODING METHODS ==========")
print("OneHotEncoder: employment_type")
print("OrdinalEncoder: education_level")
print("Education Order: High School < Graduate < Postgraduate")


# 5. BUILD COMPLETE MACHINE LEARNING PIPELINE

loan_pipeline = Pipeline(steps=[
    ("feature_engineering", LoanFeatureEngineer()),
    ("preprocessing", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=2000,
        random_state=42
    ))
])


# 6. PIPELINE CROSS-VALIDATION

print("\n========== PIPELINE CROSS-VALIDATION ==========")

cv_scores = cross_val_score(
    loan_pipeline,
    X,
    y,
    cv=5,
    scoring="accuracy"
)

print("Cross-Validation Scores:", cv_scores)
print("Pipeline CV Accuracy:", round(cv_scores.mean(), 4))


# 7. TRAIN AND SAVE FITTED PIPELINE

print("\n========== TRAINING COMPLETE PIPELINE ==========")

loan_pipeline.fit(X, y)

pipeline_path = BASE_DIR / "outputs" / "task2_loan_pipeline.joblib"

joblib.dump(loan_pipeline, pipeline_path)

print("\n========== PIPELINE SAVED ==========")
print("Saved Pipeline Path:", pipeline_path)


# 8. LOAD PIPELINE AND PREDICT THREE NEW ROWS

loaded_pipeline = joblib.load(pipeline_path)

new_customers = pd.DataFrame({

    "age": [30, 45, 52],

    "income": [65000, 120000, 85000],

    "employment_type": [
        "Salaried",
        "Self-employed",
        "Unemployed"
    ],

    "education_level": [
        "Graduate",
        "Postgraduate",
        "High School"
    ],

    "credit_score": [720, 780, 610],

    "loan_amount": [50000, 150000, 80000],

    "interest_rate": [8.5, 7.2, 12.5],

    "loan_term": [36, 60, 48],

    "loan_date": pd.to_datetime([
        "2026-01-15",
        "2026-02-20",
        "2026-03-10"
    ]),

    "existing_debt": [10000, 25000, 30000],

    "debt_to_income": [0.20, 0.25, 0.40]

})

predictions = loaded_pipeline.predict(new_customers)

prediction_probabilities = loaded_pipeline.predict_proba(
    new_customers
)

prediction_results = new_customers.copy()

prediction_results["predicted_loan_default"] = predictions

prediction_results["default_probability"] = (
    prediction_probabilities[:, 1]
)

print("\n========== PREDICTIONS FOR THREE NEW CUSTOMERS ==========")

print(prediction_results[[
    "age",
    "income",
    "loan_amount",
    "predicted_loan_default",
    "default_probability"
]])

print("\n========== TASK 2 COMPLETED ==========")