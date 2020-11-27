import numpy as np

from sklearn.decomposition import PCA

######################################################################

class GDL():
    def __init__(self, n_neighbors=10, a=1, n_clusters=None, eps=None,
                 with_pca=True, random_state=0):
        self.n_neighbors = n_neighbors
        self.a = a
        if n_clusters is not None:
            self.n_clusters = n_clusters
            self.eps = None
        elif eps is not None:
            self.n_clusters = None
            self.eps = eps
        else:
            raise ValueError("either n_clusters or eps must be defined")
        if with_pca:
            self.pca = PCA(n_components=32, random_state=random_state)
        else:
            self.pca = None
        self.X = None
        self.sigma = None
        self.weights = None
        self.clusters = None
        self.affinity = None
        self.labels_ = None

    def fit(self, X):
        if self.pca is not None:
            self.pca.fit(X)
            self.X = self.pca.transform(X)
        else:
            self.X = X
        self.build_adjacency_matrix(self.X)
        # initialization with singleton clusters
        self.clusters = [[i] for i in range(len(self.X))]
        self.affinity = self.weights * self.weights.T
        # clusters aggregation
        while not self.check_stopping_criterion():
            # find clusters with highest affinity
            i, j = np.unravel_index(
                np.argmax(self.affinity+self.affinity.T),
                self.affinity.shape
                )
            # merge clusters Ci <- Ci + Cj
            Cj = self.clusters.pop(j)
            self.clusters[i] += Cj
            # update affinity Ci -> Ck for all k
            self.affinity[i] += self.affinity[j]
            self.affinity = np.delete(self.affinity, j, 0)
            self.affinity = np.delete(self.affinity, j, 1)
            # update affinity Ck -> Ci for all k
            Ci = self.clusters[i]
            for k in range(len(self.clusters)):
                Ck = self.clusters[k]
                Wik = self.weights[np.ix_(Ci, Ck)]
                Wki = self.weights[np.ix_(Ck, Ci)]
                self.affinity[k,i] = np.sum(np.matmul(Wik, Wki)) / len(Ci)**2
            # update diagonal element
            self.affinity[i,i] = 0
        self.generate_labels()

    def build_adjacency_matrix(self, X):
        # compute pairwise Euclidean distances
        A = np.tile(X, (len(X), 1))
        B = np.repeat(X, len(X), axis=0)
        W = np.sum((A - B)**2, axis=1)
        W = np.reshape(W, (len(X), -1))
        # select k-NN
        k = self.n_neighbors
        thresh = np.partition(W, k, axis=1)[:,k]
        W[W > thresh.reshape((-1,1))] = 0
        # compute weights
        self.sigma = self.a * np.sum(W) / len(X) / k
        weights = np.exp(-W / self.sigma)
        weights[W == 0] = 0
        self.weights = weights
    
    def check_stopping_criterion(self):
        if self.n_clusters is not None:
            stop = (len(self.clusters) == self.n_clusters)
        else:
            stop = np.all(self.affinity + self.affinity.T < self.eps)
        return stop
    
    def generate_labels(self):
        n_labels = sum([len(cluster) for cluster in self.clusters])
        self.labels_ = np.empty(n_labels)
        for label, cluster in enumerate(self.clusters):
            for i in cluster:
                self.labels_[i] = label

    def predict(self, Y, alpha=1, return_affinity=False):
        if self.pca is not None:
            Y = self.pca.transform(Y)
        # cluster indicator vectors normalized with cluster cardinality
        I = np.zeros((len(self.clusters), len(self.labels_)))
        for i, cluster in enumerate(self.clusters):
            I[i][cluster] = 1 / len(cluster)
        # weight matrix between X and Y
        A = np.repeat(self.X, len(Y), axis=0)
        B = np.tile(Y, (len(self.X), 1))
        W = np.sum((A - B)**2, axis=1)
        W = np.reshape(W, (-1, len(Y)))
        W = np.exp(-W / self.sigma)
        # indegree
        knn_thresh_in = np.ma.masked_array(
            self.weights, mask=(self.weights==0)
            ).min(axis=1, keepdims=True)
        indegree = np.matmul(I, np.where(W >= knn_thresh_in, W, 0))
        # outdegree
        k = self.n_neighbors
        knn_thresh_out = -np.partition(-W, k, axis=0)[k,:]
        outdegree = np.matmul(I, np.where(W > knn_thresh_out, W, 0))
        # affinity y to clusters
        affinity = indegree * outdegree
        # affinity clusters to y
        affinity += np.matmul(I,
            np.where(W >= knn_thresh_in, W, 0) * np.where(W > knn_thresh_out, W, 0))
        # select label with highest affinity
        affinity_thresh = np.max(self.affinity + self.affinity.T)
        labels = np.where(
            np.max(affinity, axis=0) > alpha * affinity_thresh,
            np.argmax(affinity, axis=0), -1
            )
        if return_affinity:
            return labels, np.max(affinity, axis=0)
        else:
            return labels
