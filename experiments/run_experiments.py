import matplotlib.pyplot as plt
from kmeans_algorithm.utils import generate_synthetic_data
from kmeans_algorithm.kstar_means import KStarMeans

X = generate_synthetic_data(n_clusters=3)

model = KStarMeans()
model.fit(X)

print("Znalezione klastry:", len(model.centroids))

plt.scatter(X[:, 0], X[:, 1], c=model.labels_)
plt.scatter(model.centroids[:, 0], model.centroids[:, 1], c='red', marker='X')
plt.title("K*-Means clustering")
plt.show()
