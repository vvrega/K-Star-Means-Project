import numpy as np
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from kstar_means.core import KStarMeans
from kstar_means.metrics import ARI, NMI, silhouette


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
    print("Estimated k*:", len(np.unique(labels)))
    print("ARI:", ARI(true_labels, labels))
    print("NMI:", NMI(true_labels, labels))
    if len(np.unique(labels)) > 1:
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


if __name__ == "__main__":
    run()
