# Modifications made in the algorithm implemented by <https://github.com/munikarmanish/kmeans>
# Import libraries necessary for this project
import numpy as np
from scipy.spatial import distance
import networkx as nx
from nltk.cluster import KMeansClusterer
import nltk

class BisectingKmeans:

    """Bisecting-Kmeans clustering.
    Parameters
    ----------
    max_k : int, default: 10
        Maximum number of clusters of the bisecting-kmeans algorithm.
    verbose : boolean
        Verbosity mode.    
    Parameters of the Kmeans: {tol, init, metric}. 
        See Kmeans algorithm.
    
    Attributes
    ---------
    clusters_list : 
        List of clusters.
    centroids :
        Coordinates of cluster centers.

    Notes
    -----
    Each clusters is added to the networkx as a node. The label of each node consists of the cluster sse.
    Each node contains the kmeans instance and document names.
    """
    
    def __init__(self, max_k=10, verbose=True, tol=0.01, init="kmeans++", metric="cosine"):
        self.max_k = max_k
        self.tol = tol
        self.verbose = verbose
        self.init = init
        self.metric = metric
        self.G = nx.Graph()
    
    def fit(self, data):

        """Compute k-means clustering.
        Parameters
        ----------
        data : array
            Training instances to cluster.
        """

        embedding_dim = data.shape[1] - 1
        self.clusters_list = [data, ]
        self.k = len(self.clusters_list)
        self.centroids = np.array([np.mean(self.clusters_list[i][:, 1:embedding_dim].astype(float), 0) for i in range(self.k)])
        
        if self.verbose:
            print("k={:2d}, SSE={:10.4f}, GAIN={:>10}".format(self.k, self.sse(data[:, 1:embedding_dim]), '-'))
            
        for i in range(1, self.max_k):
            sse_list = [self.sse(data[:, 1:embedding_dim]) for data in self.clusters_list]
            old_sse = np.sum(sse_list)
            new_max_sse = "{:.3f}".format(sse_list[np.argmax(sse_list)])
            data = np.array(self.clusters_list.pop(np.argmax(sse_list)))
            
            kclusterer = KMeansClusterer(2, distance = nltk.cluster.util.cosine_distance, repeats=25, avoid_empty_clusters=True)
            kmeans_cluster = kclusterer.cluster(data[:, 1:embedding_dim], assign_clusters=True)

            c1, c2 = [], []
            for j in range(len(data)):
                if kmeans_cluster[j] == 0:
                    c1.append(data[j])
                else:
                    c2.append(data[j])
            
            c1, c2 = np.matrix(c1), np.matrix(c2)
            
            self.clusters_list.append(c1)
            self.clusters_list.append(c2)
            
            sse_c1 = self.sse(c1[:, 1:embedding_dim])
            sse_c2 = self.sse(c2[:, 1:embedding_dim])
            
            # Add to node the kmeans instance and document names
            self.G.add_node("{:.3f}".format(sse_c1), kmeans=kclusterer, data=c1[:, 0])
            self.G.add_node("{:.3f}".format(sse_c2), kmeans=kclusterer, data=c2[:, 0])
            if i == 1: #First node (root)
                self.G.add_node(new_max_sse, kmeans=kclusterer, data=data[:, 0])
                self.G.add_edge("{:.3f}".format(sse_c1), new_max_sse)
                self.G.add_edge("{:.3f}".format(sse_c2), new_max_sse)
            else:
                self.G.add_edge("{:.3f}".format(sse_c1), new_max_sse)
                self.G.add_edge("{:.3f}".format(sse_c2), new_max_sse)
            
            self.k += 1
            self.centroids = np.array([np.mean(self.clusters_list[j][:, 1:embedding_dim].astype(float), 0) for j in range(self.k)])

            sse_list = [self.sse(data[:, 1:embedding_dim]) for data in self.clusters_list]
            new_sse = np.sum(sse_list)
            gain = old_sse - new_sse
            if self.verbose:
                print("k={:2d}, SSE={:10.4f}, GAIN={:10.4f}".format(self.k, new_sse, gain))
            if gain < self.tol:
                break

        return self
    
    def export_data(self, name):
        """Save the instance of bisecting-kmeans as a node in networkx.
        Parameters
        ----------
        name :
            Name of gpickle file.
        """
        nx.write_gpickle(self.G, name + ".gpickle")
    
    def sse(self, data):
        centroid = np.mean(data, 0)
        return np.sum(distance.cdist(data, centroid.reshape(1, -1), self.metric))
