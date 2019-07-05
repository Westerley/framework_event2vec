Event2Vec++: Agrupamento Hier�rquico Multivis�o de Eventos usando Embedding Spaces.

-------------------------------------------------------------------------------------------

## File: dataset.py

Convert the dataset `csv file` to a network `graph.pkl` using networkx. The network will be stored in `datasets`.

#### Parameters:

--edges

- `edges.csv` is the name of your csv file.

--sep

- `\t` delimiter of columns. Default parameter.

--output

- `graph.pkl` is the name of the network.

#### Example Usage

> `python dataset.py --edges datasets/edges.csv --output graph.pkl` or

> `python dataset.py --edges datasets/edges.csv --sep=';' --output graph.pkl`

-------------------------------------------------------------------------------------------

## File: main.py

Methods to create embeddings.

#### Parameters:

Obs: Default parameters are omitted.

--method

1. Node2vec
2. DeepWalk
3. Metapath2vec
4. Line
5. NetMF

--input 

- `graph.pkl` is the name of your network file.

--nodes

- `nodes.csv` is the name of your csv file. Only for metapath2vec.

--edges

- `edges.csv` is the name of your csv file. Only for metapath2vec.

--output

- `name_embedding.csv` is the name of your embedding file. Embedding will be stored in `embeddings`.

#### Example Usage

Implementation of paper node2vec: Scalable Feature Learning for Networks by Grover. A and Leskovec, J. [Paper](https://arxiv.org/abs/1607.00653) [code](https://github.com/aditya-grover/node2vec)

> `python main.py --method 1 --input datasets/graph.pkl --output node2vec_embedding.csv --embedding_dim 128 --p 1.5 --q 2`.

Implementation of paper DeepWalk: Online Learning of Social Representations by Perozzi B, et al. [Paper](https://arxiv.org/abs/1403.6652) [code](https://github.com/phanein/deepwalk)

> `python main.py --method 2 --input datasets/graph.pkl --output deepwalk_embedding.csv --embedding_dim 128`.

Implementation of paper metapath2vec: Scalable Representation Learning for Heterogeneous Networks by Dong, Y, et al. [Paper](https://ericdongyx.github.io/papers/KDD17-dong-chawla-swami-metapath2vec.pdf) [code](https://github.com/apple2373/metapath2vec)

> `python main.py --method 3 --input datasets/graph.pkl --output meta_embedding.csv --embedding_dim 128 --care_type 1 --epochs 100 --negative_samples 3` or

> `python main.py --method 3 --input datasets/graph.pkl --output meta_embedding.csv --embedding_dim 128 --care_type 0 --epochs 100 --negative_samples 3`.

Implementation of paper LINE: Large-scale Information Network Embedding_ by Tang, J, et al. [Paper](https://arxiv.org/abs/1503.03578) [code](https://github.com/snowkylin/line)

> `python main.py --method 4 --input datasets/graph.pkl --output line_embedding.csv --embedding_dim 128 --proximity 'first-order'` or

> `python main.py --method 4 --input datasets/graph.pkl --output line_embedding.csv --embedding_dim 128 --proximity 'second-order'`.

Implementation of paper Network Embedding as Matrix Factorization: Unifying DeepWalk, LINE, PTE, and node2vec by Qiu J, et al. [Paper](https://arxiv.org/abs/1710.02971) [code](https://github.com/xptree/NetMF)

> `python main.py --method 5 --input datasets/graph.pkl --output netmf_embedding.csv --embedding_dim 128 --small` or

> `python main.py --method 5 --input datasets/graph.pkl --output netmf_embedding.csv --embedding_dim 128 --large --window 10 --rank 1024`.
