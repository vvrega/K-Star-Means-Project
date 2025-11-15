# Przykład użycia k*-means

```python
from kmeans_algorithm.utils import generate_synthetic_data
from kmeans_algorithm.kstar_means import KStarMeans

X = generate_synthetic_data()

model = KStarMeans()
model.fit(X)

print("Liczba klastrów:", len(model.centroids))