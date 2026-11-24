from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt

data = load_wine()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

gini_model = DecisionTreeClassifier(
    criterion="gini",
    random_state=42
)
gini_model.fit(X_train, y_train)

gini_predictions = gini_model.predict(X_test)
gini_accuracy = accuracy_score(y_test, gini_predictions)

entropy_model = DecisionTreeClassifier(
    criterion="entropy",
    random_state=42
)
entropy_model.fit(X_train, y_train)

entropy_predictions = entropy_model.predict(X_test)
entropy_accuracy = accuracy_score(y_test, entropy_predictions)

comparison = pd.DataFrame({
    "Criterion": ["Gini", "Entropy"],
    "Test Accuracy": [gini_accuracy, entropy_accuracy]
})

print("\nGini vs Entropy Comparison:")
print(comparison.to_string(index=False))

# 5. Tune max_depth and min_samples_leaf
param_grid = {
    "max_depth": [2, 3, 4, 5, None],
    "min_samples_leaf": [1, 2, 4, 6]
}

grid_search = GridSearchCV(
    estimator=DecisionTreeClassifier(
        criterion="gini",
        random_state=42
    ),
    param_grid=param_grid,
    cv=5,
    scoring="accuracy"
)

grid_search.fit(X_train, y_train)

print("\nBest Parameter Combination:")
print(grid_search.best_params_)

print("\nBest Cross-Validation Accuracy:")
print(grid_search.best_score_)

plt.figure(figsize=(20, 10))

plot_tree(
    gini_model,
    feature_names=data.feature_names,
    class_names=data.target_names,
    filled=True,
    rounded=True,
    max_depth=3
)

plt.title("Decision Tree - Maximum Depth 3")
plt.tight_layout()
plt.savefig("decision_tree.png", dpi=300)
plt.show()

print("\nFirst Two Splits:")

for node_id in [0, 1, 2]:
    if gini_model.tree_.children_left[node_id] != -1:
        feature_index = gini_model.tree_.feature[node_id]
        threshold = gini_model.tree_.threshold[node_id]

        print(
            f"Node {node_id}: {data.feature_names[feature_index]} "
            f"<= {threshold:.4f}"
        )

# Explanation of First Two Splits:

# First split: The decision tree divides wine samples
# based on color_intensity <= 3.82.

# Second-level splits: The tree further divides the
# resulting groups using ash <= 3.00 and
# od280/od315_of_diluted_wines <= 3.73.


importance_table = pd.DataFrame({
    "Feature": data.feature_names,
    "Importance": gini_model.feature_importances_
})

importance_table = importance_table.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 5 Important Features:")
print(importance_table.head(5).to_string(index=False))
importance_table.head(5).to_csv("features_important.csv", index=False)
  
# Explain Gini impurity in 3 lines

# Gini impurity measures how mixed the classes are in a decision tree node.
# A lower Gini value means the node contains more samples from one class.
# A Gini value of zero means all samples in that node belong to one class.