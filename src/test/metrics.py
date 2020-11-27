import numpy as np

from itertools import product
from scipy.optimize import linear_sum_assignment

#######################################################################

def get_entropy(clusters):
    """ Compute the entropy of a clustering

    Argument:
        clusters: list(list)
    """
    n = sum([len(cluster) for cluster in clusters])
    h = sum([len(cluster) * np.log(len(cluster)/n) for cluster in clusters])
    return h


def get_nmi(clustering_a, clustering_b):
    """ Compute the Normalized Mutual Information between two clusterings

    Arguments:
        clustering_x: list(list)
    """
    nmi = 0
    if len(clustering_a) > 1 and len(clustering_b) > 1:
        n = sum([len(cluster) for cluster in clustering_a])
        if n != sum([len(cluster) for cluster in clustering_b]):
            raise ValueError("the two clusterings should have the same sample size")
        for Ca, Cb in product(clustering_a, clustering_b):
            n_intersection = len(set(Ca) & set(Cb))
            if n_intersection > 0:
                nmi += n_intersection * np.log(n * n_intersection / len(Ca) / len(Cb))
        ha = get_entropy(clustering_a)
        hb = get_entropy(clustering_b)
        nmi = nmi / np.sqrt(ha * hb)
    return nmi


def get_clustering_error(clustering_pred, clustering_true, return_map=False):
    """ Compute the clustering error

    Arguments:
        clustering_pred: list(list), proposed clustering 
        clustering_true: list(list), target clustering 
    """
    n = sum([len(cluster) for cluster in clustering_pred])
    if n != sum([len(cluster) for cluster in clustering_true]):
        raise ValueError("the two clusterings should have the same sample size")
    n_clusters = len(clustering_true)
    # linear assignment problem assign exactly one task to one worker and vice versa
    if len(clustering_pred) > n_clusters:
        clustering_true = clustering_true * (len(clustering_pred) - n_clusters + 1)
    weights = np.array(
        [[len(Ca) - len(set(Ca) & set(Cb)) for Cb in clustering_true]
            for Ca in clustering_pred]
        )
    row_ind, col_ind = linear_sum_assignment(weights)
    err = weights[row_ind, col_ind].sum() / n
    if return_map:
        mapping = {i: j % n_clusters for i, j in zip(row_ind, col_ind)}
        mapping[-1] = -1
        return err, mapping
    else:
        return err



