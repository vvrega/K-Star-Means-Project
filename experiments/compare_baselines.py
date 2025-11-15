from kstar_means.core import KStarMeans
from sklearn.cluster import KMeans
from sklearn import datasets
from kstar_means.metrics import ari, nmi, silhouette

def run():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # K*-Means
    km_star = KStarMeans(max_iter=100)
    km_star.fit(X)
    labels_star = km_star.labels_

    # klasyczny k-means — musisz podać k = liczba prawdziwych klastrów
    k = len(set(y))
    km = KMeans(n_clusters=k, random_state=0)
    labels_km = km.fit_predict(X)

    print("=== K*-Means ===")
    print("Estimated k*:", len(km_star.centroids))
    print("ARI:", ari(y, labels_star))
    print("NMI:", nmi(y, labels_star))
    print("Silhouette:", silhouette(X, labels_star))

    print("\n=== Standard K-Means ===")
    print("k:", k)
    print("ARI:", ari(y, labels_km))
    print("NMI:", nmi(y, labels_km))
    print("Silhouette:", silhouette(X, labels_km))

if __name__ == "__main__":
    run()
