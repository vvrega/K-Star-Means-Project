import numpy as np
from .utils import kmeans_init, assign_points, compute_centroids
from .mdl import mdl_cost_cluster

def try_split(X, centroids, labels):
    """
    Dla każdego klastra spróbuj rozdzielić go na dwa sub-klastry.
    Jeśli podział zmniejsza koszt MDL → zaakceptuj.
    """
    new_centroids = []
    new_labels = np.zeros_like(labels)
    current_k = len(centroids)
    next_label = 0

    for i, c in enumerate(centroids):
        pts = X[labels == i]
        if pts.shape[0] < 2:
            # za mało punktów, aby sensownie podzielić
            new_centroids.append(c)
            new_labels[labels == i] = next_label
            next_label += 1
            continue

        # zainicjuj 2 centroidy z tej grupy (np. mały k-means)
        c1, c2 = kmeans_init(pts, 2)
        labels_sub = assign_points(pts, np.array([c1, c2]))
        cent_sub = compute_centroids(pts, labels_sub)

        cost_before = mdl_cost_cluster(pts, c)
        cost_after = (mdl_cost_cluster(pts[labels_sub == 0], cent_sub[0])
                      + mdl_cost_cluster(pts[labels_sub == 1], cent_sub[1]))

        if cost_after < cost_before:
            # rzeczywiście podział się opłaca
            new_centroids.append(cent_sub[0])
            new_centroids.append(cent_sub[1])
            # etykietowanie: nadaj nowe etykiety
            new_labels[labels == i][labels_sub == 0] = next_label
            new_labels[labels == i][labels_sub == 1] = next_label + 1
            next_label += 2
        else:
            # nie opłaca się dzielić
            new_centroids.append(c)
            new_labels[labels == i] = next_label
            next_label += 1

    new_centroids = np.vstack(new_centroids)
    return new_centroids, new_labels

def try_merge(X, centroids, labels):
    """
    Spróbuj scalić każdą parę klastrów – jeśli merge zmniejsza koszt MDL, wykonaj.
    Prostą strategią: porównaj wszystkie pary.
    """
    k = len(centroids)
    if k < 2:
        return centroids, labels

    best_cost = None
    best_pair = None
    best_centroids = None
    best_labels = None

    # oblicz koszty MDL dla każdej pary
    for i in range(k):
        for j in range(i + 1, k):
            # tworzymy centroid scalenia
            merged_points = X[(labels == i) | (labels == j)]
            merged_centroid = merged_points.mean(axis=0)

            cost_i = mdl_cost_cluster(X[labels == i], centroids[i])
            cost_j = mdl_cost_cluster(X[labels == j], centroids[j])
            cost_separate = cost_i + cost_j

            cost_merged = mdl_cost_cluster(merged_points, merged_centroid)

            if best_cost is None or cost_merged < best_cost:
                best_cost = cost_merged
                best_pair = (i, j)
                # buduj nowe centroidy
                new_cent = [centroids[t] for t in range(k) if t != i and t != j]
                new_cent.append(merged_centroid)
                best_centroids = np.vstack(new_cent)

                # etykietowanie: punkty i,j → nowy klaster
                new_labels = np.copy(labels)
                new_label = max(labels) + 1
                new_labels[(labels == i) | (labels == j)] = new_label
                # przesunięcie innych etykiet, by były od 0..k-1 (opcjonalnie)
                unique = np.unique(new_labels)
                mapping = {old: new for new, old in enumerate(unique)}
                new_labels = np.array([mapping[x] for x in new_labels])
                best_labels = new_labels

    # jeśli merge zmniejsza koszt względem stanu początkowego
    old_cost = sum(mdl_cost_cluster(X[labels == t], centroids[t]) for t in range(k))
    if best_cost is not None and best_cost < old_cost:
        return best_centroids, best_labels
    else:
        return centroids, labels
