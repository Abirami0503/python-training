from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
import pandas as pd

iris = load_iris()

x = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=200)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print("Actual values:", y_test)
print("Predicted values:", y_pred)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

f1 = f1_score(y_test, y_pred, average='weighted')
print("F1 Score:", f1)

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(x)

print("K-Means Clusters:")
print(clusters)

print("Dataset shape:", x.shape)
print("Target shape:", y.shape)
print("Species names:", iris.target_names)
print("Predicted values:", y_pred)
print("Training features:", x_train.shape)
print("Testing features:", x_test.shape)
print("Training labels:", y_train.shape)
print("Testing labels:", y_test.shape)

comparison = pd.crosstab(y, clusters, rownames=["Actual Species"], colnames=["K-Means Clusters"])

print("\nCluster Comparison:")
print(comparison)


