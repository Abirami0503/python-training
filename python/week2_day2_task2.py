import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import pandas as pd

X, y = make_blobs(
    n_samples=300,
    centers=4,
    random_state=7
)

print("Dataset shape:", X.shape)

inertia_values = []

for k in range(2, 9):
    kmeans = KMeans(
        n_clusters=k,
        random_state=7,
        n_init=10
    )
    kmeans.fit(X)
    inertia_values.append(kmeans.inertia_)

plt.figure(figsize=(7, 5))
plt.plot(range(2, 9), inertia_values, marker='o')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.show()

silhouette_scores = []

for k in range(2, 9):
    kmeans = KMeans(
        n_clusters=k,
        random_state=7,
        n_init=10
    )
    labels = kmeans.fit_predict(X)

    score = silhouette_score(X, labels)
    silhouette_scores.append(score)

    print(f"K = {k}, Silhouette Score = {score:.4f}")

best_k = range(2, 9)[
    silhouette_scores.index(max(silhouette_scores))
]

print("\nBest K:", best_k)

final_model = KMeans(
    n_clusters=best_k,
    random_state=7,
    n_init=10
)

final_labels = final_model.fit_predict(X)

plt.figure(figsize=(7, 5))

plt.scatter(
    X[:, 0],
    X[:, 1],
    c=final_labels,
    cmap='viridis',
    s=30
)

plt.scatter(
    final_model.cluster_centers_[:, 0],
    final_model.cluster_centers_[:, 1],
    color='red',
    marker='X',
    s=200,
    label='Centroids'
)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("K-Means Final Clusters")
plt.legend()
plt.show()

customer_data = {
    "Annual Income": [15, 18, 20, 22, 25, 60, 65, 70, 85, 90],
    "Spending Score": [80, 75, 85, 20, 25, 60, 65, 30, 90, 20]
}

customers = pd.DataFrame(customer_data)

print("\nMall Customers Dataset:")
print(customers)

X_customers = customers[["Annual Income", "Spending Score"]]

customer_model = KMeans(
    n_clusters=4,
    random_state=7,
    n_init=10
)

customers["Cluster"] = customer_model.fit_predict(X_customers)

print("\nCustomer Segments:")
print(customers)

plt.figure(figsize=(8, 6))

plt.scatter(
    customers["Annual Income"],
    customers["Spending Score"],
    c=customers["Cluster"],
    cmap="viridis",
    s=100
)

plt.scatter(
    customer_model.cluster_centers_[:, 0],
    customer_model.cluster_centers_[:, 1],
    color="red",
    marker="X",
    s=200,
    label="Centroids"
)

plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("Mall Customer Segmentation")
plt.legend()
plt.show()