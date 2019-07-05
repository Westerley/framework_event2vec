import pandas as pd
import networkx as nx
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--edges', type=str, help='input csv file')
    parser.add_argument('--sep', type=str, default='\t')
    parser.add_argument('--output', type=str)
    return parser.parse_args()

def get_node(node):
    return node.strip().split(':')[0]

def get_type(node):
    return node.strip().split(':')[1]

def main(args):
    
    edges = pd.read_csv(args.edges, sep=args.sep, names=['node1', 'node2', 'weight'])
    
    G = nx.Graph()

    edges['type1'] = edges['node1'].apply(get_type)
    edges['node1'] = edges['node1'].apply(get_node)
    edges['type2'] = edges['node2'].apply(get_type)
    edges['node2'] = edges['node2'].apply(get_node)
    
    for i, n in edges.iterrows():
        G.add_edge(n['node1'], n['node2'], weight=n['weight'])  
    
    for i, n in edges.iterrows():
        G.nodes[n['node1']]['type'] = n['type1']
        G.nodes[n['node2']]['type'] = n['type2'] 

    nx.write_gpickle(G, 'datasets/%s' %args.output)

if __name__ == "__main__":
    args = parse_args()
    print(args.sep)
    main(args)
    