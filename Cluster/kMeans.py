# Import libraries necessary for this project
import numpy as np
from scipy.spatial import distance
import random

class KMeans:

    """K-Means clustering.
    Parameters
    ----------
    n_clusters : int, default: 2
        The number of clusters
    tol : float, default: 0.01
        Relative tolerance with regards to inertia to declare convergence.
    max_iter : int, default: 100
        Maximum number of iterations of the k-means algorithm.
    metric : {'cosine', 'euclidean', ...}
        Compute distance between each pair of the two collections of inputs. See the link <https://docs.scipy.org/doc/scipy/reference/spatial.distance.html>
    init : {'random', 'random++', 'kmeans++'}
        Method for initialization, defaults to 'kmeans++'.
    n_init :
        Number of time the k-means algorithm will be run with different
        centroid seeds. The final results will be the best output of
        n_init consecutive runs in terms of inertia.
    verbose : boolean
        Verbosity mode.    
    
    Attributes
    ----------
    clusters : 
        Labels of each point
    centroids : 
        Coordinates of cluster centers.
    """
    
    def __init__(self, n_clusters=2, tol=0.01, max_iter=100, metric="cosine", init="kmeans++", n_init=10, verbose=True):
        self.n_clusters = n_clusters
        self.tol = tol
        self.max_iter = max_iter
        self.init = init
        self.n_init = n_init
        self.metric = metric
        self.verbose = verbose
        
    def fit(self, data):

        """Compute k-means clustering.
        Parameters
        ----------
        data : array
            Training instances to cluster.
        """
        
        self.min_sse = np.inf
        for n_init in range(1, self.n_init + 1):
            centroids = []
            clusters = np.zeros(len(data))
            centroids = self.init_centroid(data)
        
            old_sse = np.inf
            for i in range(self.max_iter):
                for i in range(len(data)):
                    distances = distance.cdist(data[i].reshape(1, -1), centroids, self.metric)
                    classification = np.argmin(distances)
                    clusters[i] = classification
                        
                for i in range(self.n_clusters):
                    points = [data[j] for j in range(len(data)) if clusters[j] == i]
                    centroids[i] = np.mean(points, axis=0)
                    
                new_sse = 0
                for i in range(self.n_clusters):
                    new_sse += np.sum([self.sse(data[j], centroids[i]) for j in range(len(data)) if clusters[j] == i])
                gain = old_sse - new_sse
                
                if gain < self.tol:
                    if new_sse < self.min_sse:
                        self.min_sse, self.clusters, self.centroids = new_sse, clusters, centroids
                    if self.verbose:
                        print("N_INIT={:2d}, SSE={:10.4f}, GAIN={:10.4f}".format(n_init, new_sse, gain))
                    break
                else:
                    old_sse = new_sse
                
        return self
    
    def init_centroid(self, data):
        if self.init == "random":
            return data[np.random.choice(len(data), self.n_clusters, replace=False)]
        if self.init == "random++":
            centroids = []
            cluster_list = [[] for i in range(self.n_clusters)]
            while len(data) != 0:
                idx_data = np.random.choice(len(data), 1, replace=False)
                idx_cluster = np.random.choice(len(cluster_list), 1, replace=False)
                cluster_list[idx_cluster[0]].append(data[idx_data])
                data = np.delete(data, idx_data, 0)

            for i in range(self.n_clusters):
                centroids.append(np.mean(cluster_list[i], axis=0)[0])
            return np.array(centroids)
        if self.init == "kmeans++":
            centroids = random.sample(list(data), 1)
            while len(centroids) < self.n_clusters:
                min_distances = [np.min(distance.cdist(data[i].reshape(1, -1), centroids, self.metric)) for i in range(len(data))]
                probs = min_distances / np.sum(min_distances)
                cumprobs = probs.cumsum() # Return the cumulative sum of the elements along a given axis.
                r = random.random()
                ind = np.where(cumprobs >= r)[0][0]
                centroids.append(data[ind])
            return np.array(centroids)
    
    def sse(self, data, cluster):
        return np.sum(distance.cdist(data.reshape(1, -1), cluster.reshape(1, -1), self.metric))