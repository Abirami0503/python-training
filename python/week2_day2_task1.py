# Week 2 - Day 2 - Task 1
# XGBoost Classification

import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Load Breast Cancer dataset
data = load_breast_cancer()
X = data.data
y = data.target

# 2. Split data into training, validation and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train,
    test_size=0.25,
    random_state=42,
    stratify=y_train
)

# 3. Try different learning rates
settings = [
    (0.3, 100),
    (0.1, 200),
    (0.03, 500)
]

results = []
models = []

for rate, trees in settings:

    model = XGBClassifier(
        n_estimators=trees,
        learning_rate=rate,
        early_stopping_rounds=20,
        eval_metric="logloss",
        random_state=42
    )

    # Train using validation data and early stopping
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    # 4. Test model performance
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    results.append([
        rate,
        model.best_iteration,
        accuracy
    ])

    models.append(model)

# 5. Display model comparison table
results_table = pd.DataFrame(
    results,
    columns=["learning_rate", "best iteration", "test accuracy"]
)

print("\nXGBoost Model Comparison:")
print(results_table.to_string(index=False))

# 6. Select the model with the lowest validation log loss
best_model = min(models, key=lambda m: m.best_score)

# 7. Print feature importance
importance_table = pd.DataFrame({
    "Feature": data.feature_names,
    "Importance": best_model.feature_importances_
})

importance_table = importance_table.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance:")
print(importance_table.to_string(index=False))

# 8. Plot feature importance
plt.figure(figsize=(10, 8))

plt.barh(
    importance_table["Feature"],
    importance_table["Importance"]
)

plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("XGBoost Feature Importance")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()