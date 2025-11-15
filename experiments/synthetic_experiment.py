import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from kstar_means.core import KStarMeans
from kstar_means.metrics import ARI, NMI, silhouette


def run_synthetic():
    X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=42)

    model = KStarMeans(k_min=2, k_max=10, max_iter=50, random_state=42)
    labels = model.fit_predict(X)

    print("True k:", len(set(y_true)))
    print("Estimated k*:", len(set(labels)))
    print("ARI:", ARI(y_true, labels))
    print("NMI:", NMI(y_true, labels))
    print("Silhouette:", silhouette(X, labels))

    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50)
    plt.scatter(model.centroids[:, 0], model.centroids[:, 1], c='red', marker='X', s=200)
    plt.title("Synthetic dataset - K*Means clustering")
    plt.show()


if __name__ == "__main__":
    run_synthetic()
