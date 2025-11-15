from kstar_means.core import KStarMeans
from kstar_means.metrics import ARI, NMI, silhouette
from sklearn.datasets import load_iris

def run():
    data = load_iris()
    X = data.data
    y_true = data.target
    true_k = len(set(y_true))
    print("True k (Iris):", true_k)

    model = KStarMeans(max_iter=50, k_min=2)  # wymusz minimum 2 klastry
    labels = model.fit_predict(X)
    estimated_k = len(set(labels))
    print("Estimated k*:", estimated_k)

    print("ARI:", ARI(y_true, labels))
    print("NMI:", NMI(y_true, labels))

    sil = silhouette(X, labels)
    if sil is not None:
        print("Silhouette:", sil)
    else:
        print("Silhouette: cannot compute for a single cluster")

    return X, y_true, labels, model

if __name__ == "__main__":
    run()
