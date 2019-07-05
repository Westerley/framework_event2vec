# Import libraries
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support 
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.preprocessing import LabelEncoder
from bisectingKmeans import BisectingKmeans
from kMeans import KMeans
from fScoreMeasure import FScoreMeasure
import os
import argparse

class PrepareData():
    
    def __init__(self, path, embedding_dim):
        self.embedding = pd.read_csv(path)
        self.embedding_dim = embedding_dim
        n = ["node"]
        columns = [n.append(i) for i in range(1, self.embedding_dim + 1)]
        self.embedding.columns = n
        self.process()
        self.total_category = len(self.embedding["category"].unique())

    def return_document(self, node):
        if str(node[-4:]) == ".txt":
            n = node.split(".")
            return n[0]
        
    def process(self):
        self.embedding["node"] = self.embedding["node"].astype(str)
        self.embedding["category"] = self.embedding["node"].map(self.return_document)
        self.embedding = self.embedding.loc[self.embedding["category"].isna() == False]
        self.data = self.embedding.iloc[:, 1:self.embedding_dim + 1].values
        self.label = self.embedding["category"].values

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='')
    parser.add_argument('--embedding_dim', type=int, default=2, required=True, help='embedding dimensions')
    parser.add_argument('--output', type=str, required=True, help='')
    parser.add_argument('--tol', type=float, default=0.01, help='')
    parser.add_argument('--max_iter', type=int, default=100, help='')
    parser.add_argument('--metric', type=str, default='cosine', help='cosine or euclidean')
    parser.add_argument('--init', type=str, default='kmeans++', help='random, random++ or kmeans++')
    parser.add_argument('--n_init', type=int, default=10, help='')
    parser.add_argument('--verbose', type=bool, default=True, help='')
    parser.add_argument('--max_k', type=int, default=10, help='')
    return parser.parse_args()

def main(args): 
    
    prepare_data = PrepareData(path=args.input, embedding_dim=args.embedding_dim)

    for i in range(1, 11):
        print("Execução: KMeans " + str(i))
        kmeans = KMeans(n_clusters=prepare_data.total_category, tol=args.tol, max_iter=args.max_iter, metric=args.metric, init=args.init, n_init=args.n_init, verbose=args.verbose)
        kmeans.fit(prepare_data.data)

        print("Execução: Bisecting-KMeans")
        bisectingkmeans = BisectingKmeans(max_k=args.max_k, verbose=args.verbose, tol=args.tol, init=args.init, metric=args.metric)
        bisectingkmeans.fit(prepare_data.embedding.values)
        bisectingkmeans.export_data(os.getcwd() + "/gpickle/" + str(i) + args.output)

        label_encoder = LabelEncoder()
        y_true = np.array(label_encoder.fit_transform(prepare_data.label))
        y_pred = np.array(kmeans.clusters) 

        print("Avaliação: Kmeans")
        print("Fscore 'micro': %.4f" %precision_recall_fscore_support(y_true, y_pred, average='micro')[2])
        print("Fscore 'macro': %.4f" %precision_recall_fscore_support(y_true, y_pred, average='macro')[2])
        print("Fscore 'weighted': %.4f" %precision_recall_fscore_support(y_true, y_pred, average='weighted')[2])
        print("NMI: %.4f" %normalized_mutual_info_score(y_true, y_pred, average_method='arithmetic'))
        print("Rand Index: %.4f" %adjusted_rand_score(y_true, y_pred))
        
        print("Avaliação: Bisecting-Kmeans")
        FScoreMeasure(prepare_data, bisectingkmeans)
        print("-----")

if __name__ == "__main__":

    args = parse_args()
    main(args)
