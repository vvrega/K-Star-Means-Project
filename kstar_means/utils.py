import numpy as np

def assign_points(X, centroids):
    distances = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
    labels = np.argmin(distances, axis=1)
    return labels

def compute_centroids(X, labels):
    unique = np.unique(labels)
    centroids = []
    for u in unique:
        pts = X[labels == u]
        if len(pts) == 0:
            continue
        centroids.append(pts.mean(axis=0))
    return np.vstack(centroids)

def kmeans_init(X, k):
    """Inicjalizacja centroidów – prosty wybór losowy lub k-means++."""
    n, d = X.shape
    # losowy wybór
    choices = X[np.random.choice(n, k, replace=False)]
    return choices if k == 1 else choices
