import numpy as np

class KStarMeans:

    def __init__(self):
        self.centroids = None
        self.labels_ = None

    def fit(self, X):
        X = np.array(X)
        n_samples = X.shape[0]

        # Start od 2 klastrów
        self.centroids = X[np.random.choice(n_samples, size=2, replace=False)]

        while True:
            old_centroids = self.centroids.copy()

            self.assign_clusters(X)
            self.update_centroids(X)

            self.maybe_split(X)
            self.maybe_merge()

            if np.allclose(old_centroids, self.centroids):
                break

    def assign_clusters(self, X):
        distances = np.linalg.norm(X[:, None] - self.centroids[None, :], axis=2)
        self.labels_ = np.argmin(distances, axis=1)

    def update_centroids(self, X):
        new_c = []
        for k in range(len(self.centroids)):
            pts = X[self.labels_ == k]
            if len(pts) > 0:
                new_c.append(pts.mean(axis=0))
            else:
                new_c.append(self.centroids[k])
        self.centroids = np.array(new_c)

    def maybe_split(self, X):
        new_centroids = []
        for i in range(len(self.centroids)):
            pts = X[self.labels_ == i]
            if len(pts) < 5:
                new_centroids.append(self.centroids[i])
                continue

            variance = np.mean(np.linalg.norm(pts - pts.mean(axis=0), axis=1))

            if variance > 1.5:
                c1 = pts.mean(axis=0)
                c2 = pts.mean(axis=0) + np.random.randn(*c1.shape) * 0.01
                new_centroids.append(c1)
                new_centroids.append(c2)
            else:
                new_centroids.append(self.centroids[i])

        self.centroids = np.array(new_centroids)

    def maybe_merge(self):
        merged = []
        used = set()

        for i in range(len(self.centroids)):
            if i in used:
                continue
            for j in range(i + 1, len(self.centroids)):
                if j in used:
                    continue

                d = np.linalg.norm(self.centroids[i] - self.centroids[j])

                if d < 0.3:
                    merged.append((self.centroids[i] + self.centroids[j]) / 2)
                    used.add(i)
                    used.add(j)
                    break
            else:
                merged.append(self.centroids[i])
                used.add(i)

        self.centroids = np.array(merged)
