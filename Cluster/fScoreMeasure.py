# Import libraries necessary for this project
import pandas as pd
from collections import defaultdict

class FScoreMeasure:

    """FScore Measure for cluster.
    Parameters
    ----------
    prepare_data : Object
        Instance of PrepareData.
    bisecting_kmeans : Object
        Instance of Bisecting-kmeans algorithm.

    Notes
    -----
    One of the limitations of evaluating the quality of a clustering solution produced by agglomerative methods is that it 
    is very sensitive on the choice of k, and the derived clustering solution may appear unnecessarily poor due to the presence 
    of a few outlier documents. For this reason, a better approach for comparing the clustering solutions produced by 
    agglomerative methods is to compare the overall set of clusters that are represented in the tree they produce. 
    One such measure is FScore measure. Given a particular class $C_r$ of size $n_r$ and a particular 
    cluster $S_i$ of size $n_i$, suppose $n_{ri}$ documents in the cluster $S_i$ belong to $C_r$, then the FScore 
    of this class and cluster is defined to be

    $F(C_r, S_i) = \frac{ 2 * R(C_r, S_i) * P(C_r, S_i) }{ R(C_r, S_i) + P(C_r, S_i) }$,

    where $R(C_r, S_i)$ is the recall value defined as $n_{ri}/n_r$, and $P(C_r, S_i)$ is the precision value 
    defined as $n_{ri}$/$n_i$. The FScore of the class $C_r$, is the maximum FScore value attained at any node 
    in the hierarchical clustering tree T. That is,

    $F(C_r) = \max_{S_i \in T} F(C_r, S_i)$.

    The FScore of the entire clustering solution is then defined to be the sum of the individual class FScore 
    weighted according to the class size.

    $FScore = \sum_{r=1}^c \frac{n_r}{n}F(C_r)$,

    where c is the total number of classes. A perfect clustering solution will be the one in which every class has 
    a corresponding cluster containing the exactly same documents in the resulting hierarchical tree, in which case the
    FScore will be one. In general, the higher the FScore values, the better the clustering solution is.
    """
    
    def __init__(self, prepare_data, bisecting_kmeans):
        self.prepare_data = prepare_data
        self.bisecting_kmeans = bisecting_kmeans
        self.f_max_category()
        self.f_score()
    
    # Quantity of documents of all categories
    def return_category(self):
        return self.prepare_data.embedding['category'].value_counts().to_dict()

    # Represented by n_{r} in the equation
    def qtd_doc_by_category(self, category):
        total_category = self.return_category()
        return total_category[category]

    # Represented by n_{i} in the equation
    def qtd_doc_in_cluster(self, cluster):
        return len(cluster)
    
    # Represented by n_{ri} in the equation
    def qtd_doc_by_category_in_cluster(self, category, cluster):
        total_category = defaultdict(int)
        for i in range(len(cluster)):
            total_category[cluster[i, self.prepare_data.embedding.shape[1] - 1]] += 1 
        return total_category[category]
    
    # Returns the cluster with the highest fscore of each categories
    # Ex: {'E131': [3, 0.323], 'E132': [2, 0.4]}    
    def f_max_category(self):
        category = self.return_category()
        self.category = [ i for i in category.keys() ]
        self.clusters = self.bisecting_kmeans.clusters_list
        self.category_max_fscore = defaultdict()
        for i in range(self.prepare_data.total_category):
            max_fscore = 0
            id_cluster = ''
            for j in range(self.bisecting_kmeans.max_k):
                n_ri = self.qtd_doc_by_category_in_cluster(self.category[i], self.clusters[j])
                n_i = self.qtd_doc_in_cluster(self.clusters[j])
                n_r = self.qtd_doc_by_category(self.category[i])
                precision = n_ri / n_i
                recall = n_ri / n_r
                fmeasure = 0
                if (precision + recall) > 0:
                    fmeasure = (2 * precision * recall) / (precision + recall)
                if fmeasure > max_fscore:
                    max_fscore = fmeasure
                    id_cluster = j
            # A contagem do identificador do cluster começa em 0
            self.category_max_fscore[self.category[i]] = [id_cluster, max_fscore]
            print("Category={}, Id-Cluster={:2d}, Fscore={:10.4f}".format(self.category[i], id_cluster, max_fscore))
    
    # Returns the sum of the categories with the highest fscore
    def f_score(self):
        n = len(self.prepare_data.embedding.values)
        fscore = 0
        for i in range(self.prepare_data.total_category):
            nr = self.qtd_doc_by_category(self.category[i])
            f_category = self.category_max_fscore[self.category[i]][1]
            fscore += (nr / n) * f_category
        print("FscoreMeasure:{:10.4f}".format(fscore))