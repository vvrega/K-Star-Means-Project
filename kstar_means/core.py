import numpy as np


class KStarMeans:
    def __init__(self, k_min=2, k_max=None, max_iter=100, tol=1e-4, random_state=None):
        self.k_min = k_min
        self.k_max = k_max
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = np.random.RandomState(random_state)
        self.centroids = None
        self.history = []

    def fit(self, X):
        n_samples, n_features = X.shape
        k = self.k_min
        self.centroids = X[self.random_state.choice(n_samples, k, replace=False)]

        for it in range(self.max_iter):
            # Assign clusters
            labels = self._assign_clusters(X)
            new_centroids = []

            # Split/Merge
            for j in range(len(self.centroids)):
                points_in_cluster = X[labels == j]
                if len(points_in_cluster) == 0:
                    continue
                var = np.var(points_in_cluster, axis=0).mean()
                if var > 0.5 and (self.k_max is None or len(self.centroids) + 1 <= self.k_max):
                    # Split centroid
                    delta = 0.01 * self.random_state.randn(points_in_cluster.shape[1])
                    new_centroids.append(self.centroids[j] + delta)
                    new_centroids.append(self.centroids[j] - delta)
                else:
                    new_centroids.append(self.centroids[j])

            new_centroids = np.array(new_centroids)

            # Check convergence
            if self.centroids.shape == new_centroids.shape:
                if np.allclose(self.centroids, new_centroids, atol=self.tol):
                    break
            self.centroids = new_centroids
            self.history.append(len(self.centroids))

        return self

    def _assign_clusters(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

    def fit_predict(self, X):
        self.fit(X)
        return self._assign_clusters(X)
