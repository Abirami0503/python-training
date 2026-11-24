import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import randint, uniform

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, roc_auc_score,
    precision_recall_curve, mean_absolute_error,
    mean_squared_error, r2_score
)

from xgboost import XGBClassifier


# 1. LOAD DATA AND SPLIT

df = pd.read_csv("data/loan_data.csv")

X = df.drop(columns=["loan_default", "loan_date"])
y = df["loan_default"]

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)


# 2. PREPROCESSING

numeric_features = X_train.select_dtypes(include=np.number).columns
categorical_features = X_train.select_dtypes(exclude=np.number).columns

preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), numeric_features),

    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical_features)
])


# 3. CLASSIFICATION MODELS

logistic_model = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

decision_tree_model = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", DecisionTreeClassifier(max_depth=5, random_state=42))
])

logistic_model.fit(X_train, y_train)
decision_tree_model.fit(X_train, y_train)


# 4. RANDOM FOREST GRID SEARCH

rf_pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", RandomForestClassifier(random_state=42))
])

rf_parameters = {
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [5, 10],
    "classifier__min_samples_split": [2, 5]
}

rf_search = GridSearchCV(
    rf_pipeline,
    rf_parameters,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

rf_search.fit(X_train, y_train)

best_rf_model = rf_search.best_estimator_

print("\nRANDOM FOREST GRID SEARCH")
print("Best Parameters:", rf_search.best_params_)
print("Best Score:", rf_search.best_score_)


# 5. XGBOOST RANDOMIZED SEARCH

xgb_pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", XGBClassifier(
        eval_metric="logloss",
        random_state=42,
        n_jobs=1
    ))
])

xgb_parameters = {
    "classifier__n_estimators": randint(100, 401),
    "classifier__max_depth": randint(3, 8),
    "classifier__learning_rate": uniform(0.01, 0.19),
    "classifier__subsample": uniform(0.7, 0.3),
    "classifier__colsample_bytree": uniform(0.7, 0.3)
}

xgb_search = RandomizedSearchCV(
    xgb_pipeline,
    xgb_parameters,
    n_iter=15,
    cv=5,
    scoring="f1",
    random_state=42,
    n_jobs=-1
)

xgb_search.fit(X_train, y_train)

best_xgb_model = xgb_search.best_estimator_

print("\nXGBOOST RANDOMIZED SEARCH")
print("Best Parameters:", xgb_search.best_params_)
print("Best Score:", xgb_search.best_score_)


# 6. CLASSIFICATION METRICS

models = {
    "Logistic Regression": logistic_model,
    "Decision Tree": decision_tree_model,
    "Random Forest": best_rf_model,
    "XGBoost": best_xgb_model
}

validation_f1 = {}

print("\nCLASSIFICATION METRICS")

for name, model in models.items():

    predictions = model.predict(X_val)

    accuracy = accuracy_score(y_val, predictions)
    precision = precision_score(y_val, predictions, zero_division=0)
    recall = recall_score(y_val, predictions, zero_division=0)
    f1 = f1_score(y_val, predictions, zero_division=0)

    validation_f1[name] = f1

    print(f"\n{name}")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("Confusion Matrix:")
    print(confusion_matrix(y_val, predictions))


print("\nAccuracy can be misleading when classes are imbalanced.")
print("A model may achieve high accuracy by predicting the majority class,")
print("while failing to identify many cases from the minority class.")

# 7. ROC CURVES FOR THREE MODELS

plt.figure(figsize=(8, 6))

for name in ["Logistic Regression", "Random Forest", "XGBoost"]:

    probabilities = models[name].predict_proba(X_val)[:, 1]

    fpr, tpr, _ = roc_curve(y_val, probabilities)
    auc = roc_auc_score(y_val, probabilities)

    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")

plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")
plt.legend()
plt.savefig("outputs/task4_roc_curves.png")
plt.show()


# 8. DECISION THRESHOLD WITH AT LEAST 75% RECALL

best_model_name = max(validation_f1, key=validation_f1.get)
best_model = models[best_model_name]

probabilities = best_model.predict_proba(X_val)[:, 1]

precisions, recalls, thresholds = precision_recall_curve(y_val, probabilities)

valid_indices = np.where(recalls[:-1] >= 0.75)[0]

best_index = valid_indices[np.argmax(precisions[valid_indices])]

selected_threshold = thresholds[best_index]

print("\nDECISION THRESHOLD")
print("Selected Model:", best_model_name)
print("Threshold:", selected_threshold)
print("Precision:", precisions[best_index])
print("Recall:", recalls[best_index])


# 9. REGRESSION METRICS AGAINST MEAN BASELINE

regression_features = [
    col for col in df.select_dtypes(include=np.number).columns
    if col not in ["loan_amount", "loan_default"]
]

X_reg = df[regression_features]
y_reg = df["loan_amount"]

X_reg_train, X_reg_temp, y_reg_train, y_reg_temp = train_test_split(
    X_reg, y_reg, test_size=0.4, random_state=42
)

X_reg_val, X_reg_test, y_reg_val, y_reg_test = train_test_split(
    X_reg_temp, y_reg_temp, test_size=0.5, random_state=42
)

regression_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", LinearRegression())
])

regression_model.fit(X_reg_train, y_reg_train)

regression_predictions = regression_model.predict(X_reg_val)

baseline_predictions = np.full(
    len(y_reg_val), y_reg_train.mean()
)

print("\nREGRESSION MODEL")

print("MAE:", mean_absolute_error(y_reg_val, regression_predictions))
print("RMSE:", np.sqrt(mean_squared_error(y_reg_val, regression_predictions)))
print("R-squared:", r2_score(y_reg_val, regression_predictions))

print("\nMEAN BASELINE")

print("MAE:", mean_absolute_error(y_reg_val, baseline_predictions))
print("RMSE:", np.sqrt(mean_squared_error(y_reg_val, baseline_predictions)))
print("R-squared:", r2_score(y_reg_val, baseline_predictions))


# 10. FINAL TEST SET EVALUATION

test_probabilities = best_model.predict_proba(X_test)[:, 1]

test_predictions = (
    test_probabilities >= selected_threshold
).astype(int)

print("\nFINAL BEST MODEL TEST RESULTS")
print("Best Model:", best_model_name)
print("Accuracy:", accuracy_score(y_test, test_predictions))
print("Precision:", precision_score(y_test, test_predictions, zero_division=0))
print("Recall:", recall_score(y_test, test_predictions, zero_division=0))
print("F1 Score:", f1_score(y_test, test_predictions, zero_division=0))
print("Confusion Matrix:")
print(confusion_matrix(y_test, test_predictions))