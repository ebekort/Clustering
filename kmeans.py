# kmeans.py
import numpy as np
from numpy.linalg import norm
from collections import defaultdict


def euclidean_distance(a, b, axis=None):
    # calculates euclidean distance between 2 vectors, a and b.
    return norm(a-b, axis=axis)


class KMeansClusterer:
    def __init__(self, n_clusters, n_feats):
        self.n_clusters = n_clusters
        self.n_feats = n_feats
        self.centroids = np.zeros((n_clusters, n_feats))

    def initialize_clusters(self, data):
        # initalize your clusters centers here with the Forgy method, i.e.
        # by assigning each cluster center to a random point in your data.
        self.centroids = data[np.random.choice(
            range(data.shape[0]), replace=False, size=self.n_clusters)]

    def assign(self, data):
        # in this function, you need to assign your data points to the nearest
        # cluster center

        assignments = np.array([min([
            (euclidean_distance(dat, self.centroids[i]), i)
            for i in range(len(self.centroids))])[1]
            for dat in data])

        return assignments

    def update(self, assignments, data):
        # in this function, you need to update your cluster centers,
        # according to the mean of the points in that cluster
        for i in range(self.n_clusters):
            self.centroids[i] = (np.mean(data[assignments == i], axis=0))

    def fit_predict(self, data):
        # Fit contains the loop where you will first call initialize_clusters()
        self.initialize_clusters(data)

        # Then call assign() and update() iteratively for 100 iterations
        for i in range(100):
            assignments = self.assign(data)
            self.update(assignments, data)

        # Return the assignments
        return assignments
