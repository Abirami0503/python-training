import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.cluster import KMeans

from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

os.makedirs("outputs", exist_ok=True)

# ------------------------------------------------------------
# 1. LOAD DATASET AND SPLIT THE DATA
# ------------------------------------------------------------

df = pd.read_csv("data/loan_data.csv")

print("Dataset Shape:", df.shape)
print("\nFirst 5 Records:")
print(df.head())

X = df.drop(columns=["loan_default", "loan_date"])
y = df["loan_default"]

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.40,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

print("\nData Split:")
print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Testing:", X_test.shape)

numeric_features = X_train.select_dtypes(
    include=np.number
).columns.tolist()

categorical_features = X_train.select_dtypes(
    exclude=np.number
).columns.tolist()

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# ------------------------------------------------------------
# 2. TRAIN FOUR CLASSIFICATION MODELS
# ------------------------------------------------------------

logistic_pipeline = Pipeline(steps=[
    ("preprocessing", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

logistic_pipeline.fit(X_train, y_train)
logistic_predictions = logistic_pipeline.predict(X_val)

decision_tree_pipeline = Pipeline(steps=[
    ("preprocessing", preprocessor),
    ("classifier", DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    ))
])

decision_tree_pipeline.fit(X_train, y_train)
tree_predictions = decision_tree_pipeline.predict(X_val)

random_forest_pipeline = Pipeline(steps=[
    ("preprocessing", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        oob_score=True,
        random_state=42,
        n_jobs=-1
    ))
])

random_forest_pipeline.fit(X_train, y_train)
forest_predictions = random_forest_pipeline.predict(X_val)

print("\nRandom Forest OOB Score:")
print(random_forest_pipeline.named_steps["classifier"].oob_score_)

xgb_preprocessor = preprocessor
X_train_transformed = xgb_preprocessor.fit_transform(X_train)
X_val_transformed = xgb_preprocessor.transform(X_val)

xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    eval_metric="logloss",
    early_stopping_rounds=10,
    random_state=42,
    n_jobs=-1
)

xgb_pipeline = Pipeline(steps=[
    ("preprocessing", preprocessor),
    ("classifier", xgb_model)
])

xgb_pipeline.fit(
    X_train,
    y_train,
    classifier__eval_set=[(X_val_transformed, y_val)],
    classifier__verbose=False
)

print("\nXGBOOST EARLY STOPPING CHECK")

xgb_classifier = xgb_pipeline.named_steps["classifier"]

print("Best Iteration:", xgb_classifier.best_iteration)
print("Best Validation Score:", xgb_classifier.best_score)
print("Early Stopping Rounds:", xgb_classifier.early_stopping_rounds)

if xgb_classifier.best_iteration < 299:
    print("Early stopping occurred successfully.")
else:
    print("Training reached all 300 estimators.")
xgb_predictions = xgb_pipeline.predict(X_val)

# ------------------------------------------------------------
# 3. MODEL COMPARISON TABLE
# ------------------------------------------------------------

results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "XGBoost"
    ],
    "Validation Score": [
        accuracy_score(y_val, logistic_predictions),
        accuracy_score(y_val, tree_predictions),
        accuracy_score(y_val, forest_predictions),
        accuracy_score(y_val, xgb_predictions)
    ]
})

print("\nMODEL COMPARISON TABLE")
print(results.to_string(index=False))

results.to_csv(
    "outputs/task3_model_comparison.csv",
    index=False
)

# ------------------------------------------------------------
# 4. LOGISTIC REGRESSION PREDICT PROBABILITIES
# ------------------------------------------------------------

probabilities = logistic_pipeline.predict_proba(X_val.iloc[:5])
predictions = logistic_pipeline.predict(X_val.iloc[:5])

probability_results = pd.DataFrame({
    "Predicted Class": predictions,
    "Probability of Class 0": probabilities[:, 0],
    "Probability of Class 1": probabilities[:, 1]
})

print("\nLOGISTIC REGRESSION: FIRST 5 PREDICTIONS")
print(probability_results)

# ------------------------------------------------------------
# 5. TOP 5 FEATURE IMPORTANCES
# ------------------------------------------------------------

feature_names = (
    random_forest_pipeline
    .named_steps["preprocessing"]
    .get_feature_names_out()
)

forest_importances = (
    random_forest_pipeline
    .named_steps["classifier"]
    .feature_importances_
)

forest_importance_table = pd.DataFrame({
    "Feature": feature_names,
    "Importance": forest_importances
}).sort_values(
    by="Importance",
    ascending=False
)

print("\nTOP 5 RANDOM FOREST FEATURES")
print(forest_importance_table.head(5).to_string(index=False))

xgb_importances = (
    xgb_pipeline
    .named_steps["classifier"]
    .feature_importances_
)

xgb_importance_table = pd.DataFrame({
    "Feature": feature_names,
    "Importance": xgb_importances
}).sort_values(
    by="Importance",
    ascending=False
)

print("\nTOP 5 XGBOOST FEATURES")
print(xgb_importance_table.head(5).to_string(index=False))

# ------------------------------------------------------------
# 6. REGRESSION: LINEAR, RIDGE AND LASSO
# ------------------------------------------------------------

regression_target = df["loan_amount"]

regression_features = [
    column for column in df.select_dtypes(include=np.number).columns
    if column not in ["loan_amount", "loan_default"]
]

X_reg = df[regression_features]
y_reg = regression_target

X_reg_train, X_reg_val, y_reg_train, y_reg_val = train_test_split(
    X_reg,
    y_reg,
    test_size=0.20,
    random_state=42
)

regression_preprocessor = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

regression_models = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.1, max_iter=10000)
}

coefficient_results = {}

for model_name, model in regression_models.items():

    regression_pipeline = Pipeline(steps=[
        ("preprocessing", regression_preprocessor),
        ("model", model)
    ])

    regression_pipeline.fit(X_reg_train, y_reg_train)

    coefficients = regression_pipeline.named_steps["model"].coef_

    coefficient_results[model_name] = coefficients

coefficient_table = pd.DataFrame(
    coefficient_results,
    index=regression_features
)

print("\nREGRESSION COEFFICIENT COMPARISON")
print(coefficient_table)

lasso_zero_count = np.sum(
    coefficient_table["Lasso"] == 0
)

print("\nNumber of zero coefficients in Lasso:")
print(lasso_zero_count)

# ------------------------------------------------------------
# 7. K-MEANS CUSTOMER SEGMENTATION
# ------------------------------------------------------------

clustering_features = [
    "income",
    "loan_amount",
    "credit_score"
]

X_cluster = df[clustering_features]

# Handle missing values and scale the data
cluster_imputer = SimpleImputer(strategy="median")
cluster_scaler = StandardScaler()

X_cluster_imputed = cluster_imputer.fit_transform(X_cluster)
X_cluster_scaled = cluster_scaler.fit_transform(X_cluster_imputed)

inertia_values = []
silhouette_values = []

for k in range(2, 9):

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    cluster_labels = kmeans.fit_predict(X_cluster_scaled)

    inertia_values.append(kmeans.inertia_)

    silhouette_values.append(
        silhouette_score(X_cluster_scaled, cluster_labels)
    )

print("\nK-MEANS RESULTS")

for k, inertia, silhouette in zip(
    range(2, 9),
    inertia_values,
    silhouette_values
):
    print(
        f"K={k}, Inertia={inertia:.2f}, "
        f"Silhouette Score={silhouette:.4f}"
    )

# ------------------------------------------------------------
# 8. ELBOW PLOT
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    range(2, 9),
    inertia_values,
    marker="o"
)

plt.title("K-Means Elbow Method")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.xticks(range(2, 9))
plt.grid(True)

plt.savefig("outputs/task3_elbow_plot.png")
plt.show()

# ------------------------------------------------------------
# 9. SILHOUETTE SCORE PLOT
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    range(2, 9),
    silhouette_values,
    marker="o"
)

plt.title("K-Means Silhouette Scores")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.xticks(range(2, 9))
plt.grid(True)

plt.savefig("outputs/task3_silhouette_plot.png")
plt.show()

# ------------------------------------------------------------
# 10. CUSTOMER SEGMENT DESCRIPTIONS
# ------------------------------------------------------------

best_k = range(2, 9)[np.argmax(silhouette_values)]

print("\nBest K based on Silhouette Score:", best_k)

final_kmeans = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10
)

df["Customer_Segment"] = final_kmeans.fit_predict(
    X_cluster_scaled
)

segment_summary = df.groupby("Customer_Segment")[
    clustering_features
].mean().round(2)

segment_summary["Customer_Count"] = (
    df["Customer_Segment"].value_counts()
)

print("\nCUSTOMER SEGMENT DESCRIPTIONS")
print(segment_summary)

for segment, row in segment_summary.iterrows():

    print(f"\nSegment {segment}:")

    print(
        f"Average income: {row['income']:.2f}, "
        f"Average loan amount: {row['loan_amount']:.2f}, "
        f"Average credit score: {row['credit_score']:.2f}, "
        f"Number of customers: {int(row['Customer_Count'])}"
    )

print("\nTASK 3 COMPLETED SUCCESSFULLY!")