import numpy as np
from typing import List, Tuple
from scipy.spatial.distance import cdist

class KStarMeans:
    def __init__(self, patience: int = 100):
        self.patience = patience

    def init_subcentroids(self, cluster: np.ndarray) -> List[np.ndarray]:

        if len(cluster) < 2:
            return [cluster.mean(axis=0), cluster.mean(axis=0)]

        # Simple initialization: pick two random points
        indices = np.random.choice(len(cluster), size=2, replace=False)
        submu1, submu2 = cluster[indices[0]], cluster[indices[1]]

        # Run a few k-means iterations
        for _ in range(5):
            # Assign points to nearest sub-centroid
            dist1 = np.linalg.norm(cluster - submu1, axis=1)
            dist2 = np.linalg.norm(cluster - submu2, axis=1)

            subc1 = cluster[dist1 < dist2]
            subc2 = cluster[dist1 >= dist2]

            if len(subc1) > 0:
                submu1 = subc1.mean(axis=0)
            if len(subc2) > 0:
                submu2 = subc2.mean(axis=0)

        return [submu1, submu2]

    def kmeans_step(self, X: np.ndarray, mu: List[np.ndarray],
                    C: List[np.ndarray], mu_s: List[List[np.ndarray]],
                    C_s: List[List[np.ndarray]]) -> Tuple:

        k = len(mu)

        # Assign each point to nearest centroid
        centroids_array = np.array(mu)
        distances = cdist(X, centroids_array)
        assignments = np.argmin(distances, axis=1)

        # Update clusters
        new_C = []
        new_mu = []
        new_mu_s = []
        new_C_s = []

        for i in range(k):
            cluster_points = X[assignments == i]

            if len(cluster_points) > 0:
                new_C.append(cluster_points)
                new_mu.append(cluster_points.mean(axis=0))

                # Update sub-centroids for this cluster
                sub_centroids = self.init_subcentroids(cluster_points)
                new_mu_s.append(sub_centroids)

                # Assign cluster points to sub-centroids
                submu1, submu2 = sub_centroids
                dist1 = np.linalg.norm(cluster_points - submu1, axis=1)
                dist2 = np.linalg.norm(cluster_points - submu2, axis=1)

                subc1 = cluster_points[dist1 < dist2]
                subc2 = cluster_points[dist1 >= dist2]
                new_C_s.append([subc1, subc2])
            else:
                # Keep empty clusters
                new_C.append(cluster_points)
                new_mu.append(mu[i])
                new_mu_s.append(mu_s[i] if i < len(mu_s) else [mu[i], mu[i]])
                new_C_s.append([np.array([]), np.array([])])

        return new_mu, new_C, new_mu_s, new_C_s

    def mdl_cost(self, X: np.ndarray, mu: List[np.ndarray],
                 C: List[np.ndarray]) -> float:

        n, d = X.shape
        k = len(mu)

        # Calculate float precision
        X_flat = X.flatten()
        unique_vals = np.unique(X_flat)
        if len(unique_vals) > 1:
            min_dist = np.min(np.diff(np.sort(unique_vals)))
            float_precision = -np.log(max(min_dist, 1e-10))
        else:
            float_precision = 1.0

        # Float cost
        X_range = np.max(X) - np.min(X)
        float_cost = X_range / max(float_precision, 1e-10)

        # Model cost
        model_cost = k * d * float_cost

        # Index cost
        idx_cost = n * np.log(max(k, 1))

        # Residual cost
        c = 0
        for i, cluster in enumerate(C):
            if len(cluster) > 0:
                c += np.sum((cluster - mu[i]) ** 2)

        residual_cost = (n * d * np.log(2 * np.pi) + c) / 2

        return model_cost + idx_cost + residual_cost

    def maybe_split(self, X: np.ndarray, mu: List[np.ndarray],
                    C: List[np.ndarray], mu_s: List[List[np.ndarray]],
                    C_s: List[List[np.ndarray]]) -> Tuple:

        best_cost_change = self.mdl_cost(X, mu, C)
        split_at = -1
        n = len(X)
        k = len(mu)

        for i in range(len(mu)):
            if len(C_s[i]) < 2 or len(C_s[i][0]) == 0 or len(C_s[i][1]) == 0:
                continue

            subc1, subc2 = C_s[i]
            submu1, submu2 = mu_s[i]

            # Calculate cost change
            cost_split = (np.sum((subc1 - submu1) ** 2) +
                          np.sum((subc2 - submu2) ** 2))
            cost_original = np.sum((C[i] - mu[i]) ** 2)
            cost_change = cost_split - cost_original + n / (k + 1)

            if cost_change < best_cost_change:
                best_cost_change = cost_change
                split_at = i

        did_split = False
        if best_cost_change < 0 and split_at >= 0:
            # Perform split
            submu1, submu2 = mu_s[split_at]
            subc1, subc2 = C_s[split_at]

            # Replace cluster at split_at with two new clusters
            new_mu = mu[:split_at] + [submu1, submu2] + mu[split_at + 1:]
            new_C = C[:split_at] + [subc1, subc2] + C[split_at + 1:]

            # Initialize sub-centroids for new clusters
            new_mu_s = mu_s[:split_at] + [self.init_subcentroids(subc1),
                                          self.init_subcentroids(subc2)] + mu_s[split_at + 1:]

            # Create sub-clusters
            new_C_s = []
            for cluster, subs in zip(new_C, new_mu_s):
                if len(cluster) > 0:
                    dist1 = np.linalg.norm(cluster - subs[0], axis=1)
                    dist2 = np.linalg.norm(cluster - subs[1], axis=1)
                    new_C_s.append([cluster[dist1 < dist2], cluster[dist1 >= dist2]])
                else:
                    new_C_s.append([np.array([]), np.array([])])

            mu, C, mu_s, C_s = new_mu, new_C, new_mu_s, new_C_s
            did_split = True

        return mu, C, mu_s, C_s, did_split

    def maybe_merge(self, X: np.ndarray, mu: List[np.ndarray],
                    C: List[np.ndarray], mu_s: List[List[np.ndarray]],
                    C_s: List[List[np.ndarray]]) -> Tuple:

        k = len(mu)
        n = len(X)

        if k <= 1:
            return mu, C, mu_s, C_s

        # Find closest pair of centroids
        centroids_array = np.array(mu)
        distances = cdist(centroids_array, centroids_array)
        np.fill_diagonal(distances, np.inf)
        i1, i2 = np.unravel_index(np.argmin(distances), distances.shape)

        # Calculate cost change
        Z = np.vstack([C[i1], C[i2]]) if len(C[i1]) > 0 and len(C[i2]) > 0 else (C[i1] if len(C[i1]) > 0 else C[i2])
        m_merged = Z.mean(axis=0)

        main_Q = np.sum((Z - m_merged) ** 2)
        sub_Q = np.sum((C[i1] - mu[i1]) ** 2) + np.sum((C[i2] - mu[i2]) ** 2)
        cost_change = main_Q - sub_Q - n / k

        if cost_change < 0:
            # Perform merge
            new_mu = [m for idx, m in enumerate(mu) if idx not in [i1, i2]] + [m_merged]
            new_C = [c for idx, c in enumerate(C) if idx not in [i1, i2]] + [Z]
            new_mu_s = [ms for idx, ms in enumerate(mu_s) if idx not in [i1, i2]] + [self.init_subcentroids(Z)]

            new_C_s = []
            for cluster, subs in zip(new_C, new_mu_s):
                if len(cluster) > 0:
                    dist1 = np.linalg.norm(cluster - subs[0], axis=1)
                    dist2 = np.linalg.norm(cluster - subs[1], axis=1)
                    new_C_s.append([cluster[dist1 < dist2], cluster[dist1 >= dist2]])
                else:
                    new_C_s.append([np.array([]), np.array([])])

            return new_mu, new_C, new_mu_s, new_C_s

        return mu, C, mu_s, C_s

    def fit(self, X: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        best_cost = np.inf
        unimproved_count = 0

        # Initialize with single cluster
        mu = [X.mean(axis=0)]
        C = [X]
        mu_s = [self.init_subcentroids(X)]

        # Initialize sub-clusters
        submu1, submu2 = mu_s[0]
        dist1 = np.linalg.norm(X - submu1, axis=1)
        dist2 = np.linalg.norm(X - submu2, axis=1)
        C_s = [[X[dist1 < dist2], X[dist1 >= dist2]]]

        iteration = 0
        while True:
            iteration += 1

            # K-means step
            mu, C, mu_s, C_s = self.kmeans_step(X, mu, C, mu_s, C_s)

            # Try to split
            mu, C, mu_s, C_s, did_split = self.maybe_split(X, mu, C, mu_s, C_s)

            if not did_split:
                # K-means step
                mu, C, mu_s, C_s = self.kmeans_step(X, mu, C, mu_s, C_s)

                # Try to merge
                mu, C, mu_s, C_s = self.maybe_merge(X, mu, C, mu_s, C_s)

            # Calculate cost
            cost = self.mdl_cost(X, mu, C)

            if cost < best_cost:
                best_cost = cost
                unimproved_count = 0
                best_mu = [m.copy() for m in mu]
                best_C = [c.copy() for c in C]
            else:
                unimproved_count += 1

            if unimproved_count >= self.patience:
                break

        return best_mu, best_C

    def predict(self, X: np.ndarray, mu: List[np.ndarray]) -> np.ndarray:
        centroids_array = np.array(mu)
        distances = cdist(X, centroids_array)
        return np.argmin(distances, axis=1)
