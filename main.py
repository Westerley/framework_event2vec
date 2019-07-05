import argparse
import os
import sys
import time
import logging
import gensim
import numpy as np
import pandas as pd
import networkx as nx

sys.path.append('./code')
from node2vec import Node2Vec
from deepwalk.deepwalk import process
from metapath2vec import Dataset, build_model, traning_op, train
import line
import netmf

logger = logging.getLogger(__name__)
LOGFORMAT = "%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s"

def parse_args():
    parser = argparse.ArgumentParser()
    # arguments for all methods
    parser.add_argument('--method', type=int, required=True, default=1)
    parser.add_argument('--input', type=str, help='input graph.pkl file from networkx')
    parser.add_argument('--output', required=True, help='Output representation file')
    parser.add_argument('--embedding_dim', type=int, default=2, help='embedding dimensions')
    # arguments for node2Vec
    parser.add_argument('--p', type=float, default=1, help='')
    parser.add_argument('--q', type=float, default=1, help='')
    # arguments for deepwalk
    parser.add_argument('--max-memory-data-size', type=int, default=1000000000, help='Size to start dumping walks to disk, instead of keeping them in memory.')
    parser.add_argument('--seed', type=int, default=0, help='seed for random walk generator.')
    parser.add_argument('--vertex-freq-degree', default=False, action='store_true', help='Use vertex degree to estimate the frequency of nodes '
                        'in the random walks. This option is faster than '
                        'calculating the vocabulary.')
    # arguments for node2Vec and deepwalk
    parser.add_argument('--undirected', type=bool, default=True)
    parser.add_argument('--walks', type=str, default=10, help='text file that has a random walk in each line. A random walk is just a seaquence of node ids separated by a space.')
    parser.add_argument('--walk_length', type=int, default=80)
    parser.add_argument('--workers', type=int, default=1)
    # arguments for node2vec, deepwalk, netmf, line e metapath2vec
    parser.add_argument('--window', type=int, default=10, help='context window size')
    # arguments for metapath2vec
    parser.add_argument('--epochs',type=int, default=2, help='number of epochs')
    parser.add_argument('--log_interval', type=int, default=-1, help='log intervals. -1 means per epoch')
    parser.add_argument('--max_keep_model', type=int, default=10, help='number of models to keep saving')
    parser.add_argument('--care_type', type=int, default=1, help='care type or not. if 1, it cares (i.e. heterogeneous negative sampling). If 0, it does not care (i.e. normal negative sampling). ')
    # arguments for line
    parser.add_argument('--batch_size', default=2)
    parser.add_argument('--proximity', default='second-order', help='first-order or second-order')
    parser.add_argument('--num_batches', type=int, default=200000)
    # argumentos for line and metapath2vec
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate')
    # argumentos for netmf and metapath2vec
    parser.add_argument('--negative_samples', type=int, default=5, help='number of negative samples')
    # arguments for netmf
    parser.add_argument("--rank", type=int, default=256, help="#eigenpairs used to approximate normalized graph laplacian.")
    parser.add_argument('--large', dest="large", action="store_true", help="using netmf for large window size")
    parser.add_argument('--small', dest="large", action="store_false", help="using netmf for small window size")
    parser.set_defaults(large=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    return parser.parse_args()

def main(args):
    if args.method == 1:
        # node2vec
        graph = Node2Vec(nx_G=args.input, undirected=args.undirected, p=args.p, q=args.q, dimention=args.embedding_dim)
        graph.preprocess_transition_probs()
        walks = graph.simulate_walks(args.walks, args.walk_length)
        walks = [list(map(str, walk)) for walk in walks]
        model = gensim.models.Word2Vec(walks, size=args.embedding_dim, window=args.window, min_count=0, sg=1, workers=args.workers, iter=1)
        model.wv.save_word2vec_format('embeddings/%s' %args.output)
        aux = pd.read_csv('embeddings/%s' %args.output, skiprows=1, sep=' ')
        aux.to_csv('embeddings/%s' %args.output, sep=',', index=False)
    elif args.method == 2:
        # DeepWalk
        numeric_level = getattr(logging, "INFO", None)
        logging.basicConfig(format=LOGFORMAT)
        logger.setLevel(numeric_level)
        process(args)
    elif args.method == 3:
        # Metapath2vec
        args.log = os.getcwd() + '/code/log'
        if os.path.isdir(args.log):
            print("%s already exist. are you sure to override? Ok, I'll wait for 5 seconds. Ctrl-C to abort."%args.log)
            time.sleep(5)
            os.system('rm -rf %s/'%args.log)
        else:
            os.makedirs(args.log)
            print("made the log directory",args.log)

        G = nx.read_gpickle(args.input) 
        edges = pd.DataFrame(list(G.edges()), columns=['node1', 'node2'])
        nodes = pd.DataFrame(sorted(list(G.nodes(data=True)), reverse=True), columns=['node', 'type'])
        nodes['type'] = nodes['type'].map(lambda t: (dict(t).get('type')))

        dataset=Dataset(nodes=nodes, edges=edges, window_size=args.window)
        center_node_placeholder,context_node_placeholder,negative_samples_placeholder,loss = build_model(BATCH_SIZE=1,VOCAB_SIZE=len(dataset.nodeid2index),EMBED_SIZE=args.embedding_dim,NUM_SAMPLED=args.negative_samples)
        optimizer = traning_op(loss,LEARNING_RATE=args.learning_rate)
        train(center_node_placeholder,context_node_placeholder,negative_samples_placeholder,loss,dataset,optimizer,NUM_EPOCHS=args.epochs,BATCH_SIZE=1,NUM_SAMPLED=args.negative_samples,care_type=args.care_type,LOG_DIRECTORY=args.log,LOG_INTERVAL=args.log_interval,MAX_KEEP_MODEL=args.max_keep_model,OUTPUT=args.output) 
    elif args.method == 4:
        # Line
        line.train(args)
    elif args.method == 5:
        # Netmf
        if args.large:
            netmf.netmf_large(args)
        else:
            netmf.netmf_small(args)

if __name__ == "__main__":
    args = parse_args()
    main(args)
