import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from kstar_means.core import KStarMeans
from kstar_means.metrics import ARI, NMI, silhouette

def run():
    # -----------------------
    # Generowanie danych syntetycznych
    # -----------------------
    n_samples = 300
    true_k = 7
    X, true_labels = make_blobs(n_samples=n_samples, centers=true_k, cluster_std=1.0, random_state=42)

    print("Synthetic data - true k:", true_k)

    # -----------------------
    # K*-Means
    # -----------------------
    kstar_model = KStarMeans()
    mu, C = kstar_model.fit(X)

    # Predykcja dla nowych danych
    kstar_labels = kstar_model.predict(X, mu)


    print("\n--- K*-Means ---")
    print("Estimated k*:", len(np.unique(kstar_labels)))
    print("ARI:", ARI(true_labels, kstar_labels))
    print("NMI:", NMI(true_labels, kstar_labels))
    if len(np.unique(kstar_labels)) > 1:
        print("Silhouette:", silhouette(X, kstar_labels))
    else:
        print("Silhouette: Cannot compute (only 1 cluster)")

    # -----------------------
    # Klasyczny K-Means
    # -----------------------
    kmeans_model = KMeans(n_clusters=10, random_state=42)
    kmeans_labels = kmeans_model.fit_predict(X)

    print("\n--- Classical K-Means ---")
    print("Estimated k:", true_k)
    print("ARI:", ARI(true_labels, kmeans_labels))
    print("NMI:", NMI(true_labels, kmeans_labels))
    print("Silhouette:", silhouette(X, kmeans_labels))

    # -----------------------
    # Wizualizacja wyników
    # -----------------------
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], c=kstar_labels, cmap='tab10', s=50)
    plt.title("K*-Means")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")

    plt.subplot(1, 2, 2)
    plt.scatter(X[:, 0], X[:, 1], c=kmeans_labels, cmap='tab10', s=50)
    plt.title("Classical K-Means")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run()
