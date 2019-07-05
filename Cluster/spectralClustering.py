import networkx as nx
from sklearn.cluster import SpectralClustering
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support 
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.preprocessing import LabelEncoder
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='')
    return parser.parse_args()

def return_document(node):
    if node[-4:] == ".txt":
        n = node.split(".")
        return n[0]    

def main(args):

    G = nx.read_gpickle(path=args.input)

    data = {'node': G.nodes()}

    df = pd.DataFrame(data)
    df['node'] = df['node'].astype(str)
    df['node_document'] = df['node'].map(return_document)
    n_clusters = len(df['node_document'].unique()) - 1

    adjacency_matrix = nx.adjacency_matrix(G, nodelist = G.nodes())
    for i in range(1, 11):
        sc = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', assign_labels='kmeans').fit(adjacency_matrix.todense())
        df['category'] = sc.labels_
        df_teste = df.loc[df['node_document'].isna() == False]
 
        label_encoder = LabelEncoder()
        y_true = np.array(label_encoder.fit_transform(df_teste['node_document']))
        y_pred = np.array(df_teste['category']) 

        print("Avaliação: " + str(i))
        print("Fscore 'micro': %.4f" %precision_recall_fscore_support(y_true, y_pred, average='micro')[2])
        print("Fscore 'macro': %.4f" %precision_recall_fscore_support(y_true, y_pred, average='macro')[2])
        print("Fscore 'weighted': %.4f" %precision_recall_fscore_support(y_true, y_pred, average='weighted')[2])
        print("NMI: %.4f" %normalized_mutual_info_score(y_true, y_pred, average_method='arithmetic'))
        print("Rand Index: %.4f" %adjusted_rand_score(y_true, y_pred))
        print("-----")

if __name__ == "__main__":
    
    args = parse_args()
    main(args)