from sklearn.datasets import make_blobs

def generate_synthetic_data(n_samples=400, n_clusters=3, n_features=2):
    X, _ = make_blobs(
        n_samples=n_samples,
        centers=n_clusters,
        n_features=n_features,
        random_state=42
    )
    return X
