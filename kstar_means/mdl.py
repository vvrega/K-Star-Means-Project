import numpy as np

def mdl_cost(X, labels, centroids):
    """
    Oblicza koszt MDL dla całego zbioru:
    - koszt indeksu (przypisanie etykiet)
    - koszt reszt (odległości od centroidów)
    """
    n, d = X.shape
    k = len(centroids)

    # koszt indeksu: ile bitów na zakodowanie etykiety klastra
    # zakładamy kodowanie optymalne: - sum_i (n_i / n) * log2(n_i / n) * n
    unique, counts = np.unique(labels, return_counts=True)
    probs = counts / n
    # entropia * n
    index_cost = - np.sum(counts * np.log2(probs + 1e-12))

    # koszt reszt: zakładamy, że punkty są kodowane jako odchylenia od centroidu
    # zakładamy że odchylenia ~ N(0, sigma^2) w każdej klasie — uproszczenie
    residual_cost = 0.0
    for i, c in enumerate(centroids):
        pts = X[labels == i]
        if pts.shape[0] == 0:
            continue
        # wariancja od centroidu
        var = np.var(pts - c, axis=0)
        # dla każdego wymiaru: koszt zakodowania wariancji
        # uproszczenie: użyjemy formuły ~ sum(log(var))
        residual_cost += pts.shape[0] * np.sum(np.log2(var + 1e-12))

    # suma kosztów
    total = index_cost + residual_cost
    return total

def mdl_cost_cluster(X_cluster, centroid):
    """
    Koszt MDL tylko dla jednego klastra (przydatne przy split/merge).
    """
    pts = X_cluster
    n, d = pts.shape
    if n == 0:
        return 0.0

    # indeksowy koszt dla punktów w klastrze – ale tu zakładamy,
    # że znamy klaster, więc przypisanie nie kodujemy (lub można przyjąć 0)
    index_cost = 0.0

    var = np.var(pts - centroid, axis=0)
    residual_cost = n * np.sum(np.log2(var + 1e-12))
    return index_cost + residual_cost
