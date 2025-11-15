# experiments/realdata_experiment.py

import numpy as np
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from kstar_means.core import KStarMeans
from kstar_means.metrics import ARI, NMI, silhouette
import matplotlib.pyplot as plt

def run():
    # Wczytanie danych
    data = load_iris()
    X = data.data
    true_labels = data.target
    true_k = len(np.unique(true_labels))

    print("True k (Iris):", true_k)

    # -----------------------
    # K*-Means
    # -----------------------
    model = KStarMeans(max_iter=50)
    labels = model.fit_predict(X)

    print("\n--- K*-Means ---")
    estimated_k_star = len(np.unique(labels))
    print("Estimated k*:", estimated_k_star)
    print("ARI:", ARI(true_labels, labels))
    print("NMI:", NMI(true_labels, labels))
    if estimated_k_star > 1:
        print("Silhouette:", silhouette(X, labels))
    else:
        print("Silhouette: Cannot compute (only 1 cluster)")

    # -----------------------
    # Klasyczny K-Means
    # -----------------------
    kmeans_model = KMeans(n_clusters=true_k, random_state=42)
    kmeans_labels = kmeans_model.fit_predict(X)

    print("\n--- Classical K-Means ---")
    print("Estimated k:", true_k)
    print("ARI:", ARI(true_labels, kmeans_labels))
    print("NMI:", NMI(true_labels, kmeans_labels))
    print("Silhouette:", silhouette(X, kmeans_labels))

    # Rysowanie wykresów
    plt.figure(figsize=(12, 5))  # większe okno, obok siebie

    # Lewy: K*-Means
    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50)
    plt.title("K*-Means Clustering")
    plt.xlabel('Sepal length')
    plt.ylabel('Petal length')

    # Prawy: Klasyczny K-Means
    plt.subplot(1, 2, 2)
    plt.scatter(X[:, 0], X[:, 1], c=kmeans_labels, cmap='viridis', s=50)
    plt.title("Classical K-Means Clustering")
    plt.xlabel('Sepal length')
    plt.ylabel('Petal length')

    plt.tight_layout()  # dopasowuje elementy, żeby się nie nakładały
    plt.show()

if __name__ == "__main__":
    run()
