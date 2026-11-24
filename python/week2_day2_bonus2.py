# Week 3 - Day 2
# Bonus Challenge - Random Forest vs Extra Trees

# Import required libraries
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

# 1. Load the Wine dataset
data = load_wine()
X = data.data
y = data.target

# 2. Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Train Random Forest
rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_predictions)

# 4. Train Extra Trees Classifier
et_model = ExtraTreesClassifier(
    n_estimators=200,
    random_state=42
)

et_model.fit(X_train, y_train)

et_predictions = et_model.predict(X_test)
et_accuracy = accuracy_score(y_test, et_predictions)

# 5. Compare both models
comparison = pd.DataFrame({
    "Model": ["Random Forest", "Extra Trees"],
    "Test Accuracy": [rf_accuracy, et_accuracy]
})

print("Random Forest vs Extra Trees Comparison:")
print(comparison.to_string(index=False))

# 6. Identify the better model
if rf_accuracy > et_accuracy:
    print("\nRandom Forest performed better.")
elif et_accuracy > rf_accuracy:
    print("\nExtra Trees performed better.")
else:
    print("\nBoth models achieved the same accuracy.")