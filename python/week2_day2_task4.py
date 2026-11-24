# Week 3 - Day 2
# Task 2 - Random Forest

# Import required libraries
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the Wine dataset
data = load_wine()
X = data.data
y = data.target

# 2. Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Train a single Decision Tree for comparison
tree_model = DecisionTreeClassifier(
    random_state=42
)
tree_model.fit(X_train, y_train)

tree_predictions = tree_model.predict(X_test)
tree_accuracy = accuracy_score(y_test, tree_predictions)

# 4. Train Random Forest with 200 estimators and OOB score
rf_model = RandomForestClassifier(
    n_estimators=200,
    oob_score=True,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_predictions)

print("Single Decision Tree Test Accuracy:", tree_accuracy)
print("Random Forest Test Accuracy:", rf_accuracy)
print("Random Forest OOB Score:", rf_model.oob_score_)

# 5. Try different numbers of estimators
estimators = [10, 50, 100, 300]
accuracies = []

for n in estimators:
    model = RandomForestClassifier(
        n_estimators=n,
        random_state=42
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    accuracies.append(accuracy)

    print(f"Estimators: {n}, Test Accuracy: {accuracy:.4f}")

# 6. Plot accuracy against number of estimators
plt.figure(figsize=(8, 5))
plt.plot(estimators, accuracies, marker="o")
plt.xlabel("Number of Estimators")
plt.ylabel("Test Accuracy")
plt.title("Random Forest Accuracy vs Number of Estimators")
plt.xticks(estimators)
plt.grid(True)
plt.tight_layout()
plt.show()

# 7. Plot feature importances
importance_table = pd.DataFrame({
    "Feature": data.feature_names,
    "Importance": rf_model.feature_importances_
})

importance_table = importance_table.sort_values(
    by="Importance",
    ascending=False
)

print("\nRandom Forest Feature Importances:")
print(importance_table.to_string(index=False))

plt.figure(figsize=(10, 6))
plt.barh(
    importance_table["Feature"],
    importance_table["Importance"]
)
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Random Forest Feature Importance")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("random_forest_feature_importance.png", dpi=300)
plt.show()

# 8. Written explanation

# Bootstrap aggregation (bagging) trains multiple models on
# different bootstrap samples of the training dataset.
# Random Forest combines predictions from multiple decision trees.
# It overfits less than a single tree because combining trees
# reduces prediction variance and makes results more stable.