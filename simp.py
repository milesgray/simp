"""
pyflann needs to be converted from python2 to python3. Here is an example if you are using an AWS sagemaker notebook instance:
`pip install 2to3`
`2to3 -w ~/anaconda3/envs/pytorch_p36/lib/python3.6/site-packages/pyflann`
"""
import time
import warnings
import argparse
import numpy as np
from sklearn import metrics
import scipy.sparse as sparse
try:
    import pyflann
    use_flann = True
except Exception as e:
    use_flann = False    
    pass

def get_rank(mat, dist_metric='cosine'):
    dists = metrics.pairwise.pairwise_distances(mat, mat, metric=dist_metric)
    np.fill_diagonal(dists, 1000.0)
    return np.argmin(dists, axis=1)

def get_rank_flann(mat):
    result, _ = pyflann.FLANN().nn(mat, mat, num_neighbors=2, algorithm="kdtree", trees=8, checks=128)
    return result[:, 1]

def get_adj(mat, dist_metric='cosine', flann_cutoff=100000, log=print):
    mat = mat.astype(np.float32)
    size = mat.shape[0]
    if size <= flann_cutoff:        
        rank = get_rank(mat, dist_metric=dist_metric)
    else:
        if not use_flann:
            rank = get_rank(mat, dist_metric=dist_metric)
        else:     
            rank = get_rank_flann(mat)
    data = np.ones_like(rank, dtype=np.float32)
    row = np.arange(0, size)
    col = rank
    A = sparse.csr_matrix((data, (row, col)), shape=(size, size))
    A = A + sparse.eye(size, dtype=np.float32, format='csr')
    A = np.dot(A, A.T).tolil()
    A.setdiag(0)
    return A

def get_clusters(adj, min_sim=None):
    if min_sim is not None: adj[np.where(([] * adj.toarray()) > min_sim)] = 0
    return sparse.csgraph.connected_components(csgraph=adj, directed=True, connection='weak', return_labels=True)

def cluster(data, dist_metric='cosine', min_sim=None, flann_cutoff=100000, verbose=True, log=print):
    """
        Simple Integer Measured Partition Clustering
    """
    num_clust, clusters = get_clusters(get_adj(data, dist_metric=dist_metric, flann_cutoff=flann_cutoff), min_sim)
    if verbose: log(f'{num_clust} clusters')
    return clusters
