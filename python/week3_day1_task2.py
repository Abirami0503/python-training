import pandas as pd

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

print("Dataset shape:", X.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

baseline = DummyRegressor(strategy="mean")
baseline.fit(X_train, y_train)

baseline_pred = baseline.predict(X_test)

models = {
    "Linear Regression": y_pred,
    "Baseline": baseline_pred
}

results = []

for model_name, predictions in models.items():
    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    results.append([model_name, mae, rmse, r2])

metrics_table = pd.DataFrame(
    results,
    columns=["Model", "MAE", "RMSE", "R2"]
)

print("\nFinal Metrics Table:")
print(metrics_table.round(2).to_string(index=False))