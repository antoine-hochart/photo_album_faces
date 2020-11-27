import sys
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '..')))

from time import time

from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import DBSCAN, OPTICS, AgglomerativeClustering

from pkg.gdl import GDL
from pkg.utils import get_clusters

from test.metrics import get_nmi, get_clustering_error
from test.utils import load_lfw_embeddings

#######################################################################

def cluster_comparison(X, clusters_true):
    X = PCA(n_components=32).fit_transform(X)
    connectivity = kneighbors_graph(X, n_neighbors=20, mode='distance',
                                    include_self=False)
    connectivity = 0.5 * (connectivity + connectivity.T)
    dbscan = DBSCAN(eps=0.75)
    # optics = OPTICS(min_samples=5, xi=0.05, min_cluster_size=0.05)
    ward = AgglomerativeClustering(n_clusters=len(clusters_true), linkage='ward',
        connectivity=connectivity)
    ## gdl = GDL(n_neighbors=40, eps=0.010, with_pca=False)
    gdl = GDL(n_neighbors=12, eps=0.075, with_pca=False)
    clustering_algorithms = (
        ('DBSCAN', dbscan),
        # ('OPTICS', optics),
        ('Ward', ward),
        ('GDL', gdl)
        )
    print()
    print("{:<7} {:<8} {:<8} {:<4} {:8}".format(
        'algo', 'NMI', 'Err', '|C|', 'time (s)'
        ))
    print("-"*39)
    for name, algorithm in clustering_algorithms:
        t0 = time()
        algorithm.fit(X)
        t1 = time()
        clusters_pred = get_clusters(algorithm.labels_)
        nmi = get_nmi(clusters_pred, clusters_true)
        err = get_clustering_error(clusters_pred, clusters_true)
        print("{:<7} {:<8.5f} {:<8.5f} {:<4d} {:<8.2f}".format(
            name, nmi, err, len(clusters_pred), t1 - t0
            ))

#######################################################################

if __name__ == "__main__":
    ## X, clusters_true = load_lfw_embeddings(n_familiars=7, n_strangers=14,
    ##     range_familiars=(50, 200), range_strangers=(1,3), seed=0)
    X, clusters_true = load_lfw_embeddings(n_familiars=40, n_strangers=10,
        range_familiars=(10, 20), range_strangers=(1,2), seed=0)
    print()
    print("Clustering algorithms comparison")
    cluster_comparison(X, clusters_true)