from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

def ARI(true_labels, pred_labels):
    return adjusted_rand_score(true_labels, pred_labels)

def NMI(true_labels, pred_labels):
    return normalized_mutual_info_score(true_labels, pred_labels)

def silhouette(X, labels):
    if len(set(labels)) < 2:
        return float('nan')  # nie można policzyć silhouette dla 1 klastra
    return silhouette_score(X, labels)
