import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent

data_path = BASE_DIR / "data" / "loan_data.csv"

df = pd.read_csv(data_path, parse_dates=["loan_date"])

print("\n========== DATASET LOADED ==========")
print("Dataset Shape:", df.shape)
print(df.head())

X = df.drop(columns=["loan_default", "loan_date"])
y = df["loan_default"]

numeric_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.40,
    random_state=42,
    stratify=y
)

# Second split: divide temporary data equally into validation and test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

print("\n========== DATA SPLIT ==========")
print("Training Set:", X_train.shape, y_train.shape)
print("Validation Set:", X_val.shape, y_val.shape)
print("Test Set:", X_test.shape, y_test.shape)

print("\nTarget Distribution in Each Set:")
print("Training:\n", y_train.value_counts(normalize=True))
print("Validation:\n", y_val.value_counts(normalize=True))
print("Test:\n", y_test.value_counts(normalize=True))

numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

depth_values = [1, 3, 5, 10, None]

train_accuracies = []
validation_accuracies = []

print("\n========== DECISION TREE OVERFITTING ==========")

for depth in depth_values:

    tree_model = Pipeline(steps=[
        ("preprocessing", preprocessor),
        ("classifier", DecisionTreeClassifier(
            max_depth=depth,
            random_state=42
        ))
    ])

    tree_model.fit(X_train, y_train)

    train_accuracy = tree_model.score(X_train, y_train)
    validation_accuracy = tree_model.score(X_val, y_val)

    train_accuracies.append(train_accuracy)
    validation_accuracies.append(validation_accuracy)

    print(
        f"Max Depth: {depth}, "
        f"Training Accuracy: {train_accuracy:.4f}, "
        f"Validation Accuracy: {validation_accuracy:.4f}"
    )


# Plot training and validation accuracy
plt.figure(figsize=(9, 5))

depth_labels = [str(depth) for depth in depth_values]

plt.plot(
    depth_labels,
    train_accuracies,
    marker="o",
    label="Training Accuracy"
)

plt.plot(
    depth_labels,
    validation_accuracies,
    marker="o",
    label="Validation Accuracy"
)

plt.xlabel("Decision Tree Maximum Depth")
plt.ylabel("Accuracy")
plt.title("Decision Tree: Training vs Validation Accuracy")
plt.legend()
plt.grid(True)

plt.tight_layout()

output_dir = BASE_DIR / "outputs"
output_dir.mkdir(exist_ok=True)

plt.savefig(output_dir / "task1_overfitting.png")
plt.show()

logistic_pipeline = Pipeline(steps=[
    ("preprocessing", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=2000,
        random_state=42
    ))
])

cv_scores = cross_val_score(
    logistic_pipeline,
    X_train,
    y_train,
    cv=5,
    scoring="accuracy"
)

print("\n========== LOGISTIC REGRESSION 5-FOLD CV ==========")

print("Individual Fold Scores:", cv_scores)
print("Mean Accuracy:", round(cv_scores.mean(), 4))
print("Standard Deviation:", round(cv_scores.std(), 4))

numeric_data = df[numeric_features].copy()

bad_imputer = SimpleImputer(strategy="median")
bad_scaler = StandardScaler()

X_bad_imputed = bad_imputer.fit_transform(numeric_data)
X_bad_scaled = bad_scaler.fit_transform(X_bad_imputed)

X_bad_train, X_bad_val, y_bad_train, y_bad_val = train_test_split(
    X_bad_scaled,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n========== DATA LEAKAGE DEMONSTRATION ==========")
print("Incorrect Method - Scaler fitted using the entire dataset.")
print("Training data mean after scaling:",
      np.round(X_bad_train.mean(axis=0)[:5], 4))

X_correct_train, X_correct_val, y_correct_train, y_correct_val = train_test_split(
    numeric_data,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

correct_imputer = SimpleImputer(strategy="median")
correct_scaler = StandardScaler()

X_correct_train_imputed = correct_imputer.fit_transform(X_correct_train)
X_correct_val_imputed = correct_imputer.transform(X_correct_val)
X_correct_train_scaled = correct_scaler.fit_transform(
    X_correct_train_imputed
)

X_correct_val_scaled = correct_scaler.transform(
    X_correct_val_imputed
)

print("\nCorrect Method - Scaler fitted using training data only.")
print("Training data mean after scaling:",
      np.round(X_correct_train_scaled.mean(axis=0)[:5], 4))

print("Validation data mean after scaling:",
      np.round(X_correct_val_scaled.mean(axis=0)[:5], 4))

print("\n========== DATA LEAKAGE EXPLANATION ==========")

print("1. Data leakage occurs when information from validation or test data is used during training.")
print("2. Scaling before splitting allows the scaler to learn statistics from the entire dataset.")
print("3. This can make model evaluation unreliable because future data influences preprocessing.")
print("4. To prevent leakage, split first and fit preprocessing steps only on training data.")

print("\n========== SUPERVISED AND UNSUPERVISED LEARNING ==========")

ml_examples = {
    "Spam Detection": "Supervised Learning",
    "Customer Segmentation": "Unsupervised Learning",
    "House Price Prediction": "Supervised Learning",
    "Anomaly Detection": "Unsupervised Learning"
}

for problem, learning_type in ml_examples.items():
    print(f"{problem}: {learning_type}")

print("\n========== SEVEN STEPS IN MACHINE LEARNING WORKFLOW ==========")

workflow_steps = [
    "1. Define the problem and identify the objective.",
    "2. Collect relevant data from suitable sources.",
    "3. Explore the data and handle missing or incorrect values.",
    "4. Preprocess the data and perform feature engineering.",
    "5. Split the dataset into training, validation, and test sets.",
    "6. Train models and tune their hyperparameters.",
    "7. Evaluate the final model, deploy it, and monitor its performance."
]

for step in workflow_steps:
    print(step)

results = pd.DataFrame({
    "Max Depth": depth_labels,
    "Training Accuracy": train_accuracies,
    "Validation Accuracy": validation_accuracies
})

results.to_csv(output_dir / "task1_decision_tree_results.csv", index=False)

print("\n========== TASK 1 COMPLETED ==========")
print("Overfitting plot saved in outputs folder.")
print("Decision Tree results saved in outputs folder.")