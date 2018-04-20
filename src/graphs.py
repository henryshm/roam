from collections import defaultdict
from heapq import heappush, heappop

import numpy as np
from scipy.sparse import csr_matrix

from sklearn.utils.graph_shortest_path import graph_shortest_path
from sklearn.manifold import TSNE


class ConnectionTally:

    def __init__(self, N):
        self._N = N
        self._conn = defaultdict(float)

    def connect(self, i, j, w=1.0):
        N = self._N
        assert 0 <= i < N and 0 <= j < N, "index out of bounds"
        assert i != j, "connection to self not supported"
        key = (i, j) if i < j else (j, i)
        self._conn[key] += w

    def to_sparse_matrix(self):
        N = self._N
        conn = self._conn
        xrows = [k[0] for k in conn.keys()]
        xcols = [k[1] for k in conn.keys()]
        xvals = list(conn.values())
        mat = csr_matrix((xvals, (xrows, xcols)), shape=(N,N))
        # make symmetric
        return mat + mat.T


def get_prob_matrix(weights):
    """
    Return a matrix of symmetrized joint probabilities for each weighted connection.
    The resulting probabilities sum to 1.0.
    """

    N, M = weights.shape
    assert N == M, "a square matrix is required"

    # compute sum of weights for each node
    wsum = np.sum(weights, 0)

    # avoid divide-by-zero for nodes that have no weight
    wsum[wsum == 0.0] = 1.0

    # compute symmetrized probabilities
    prob = (weights / wsum) + (weights / wsum.T)
    prob /= 2*N

    return prob


def prob_to_dist_func(prob):
    """
    Return a matrix of distances based the given probabilities matrix.
    """

    N, M = prob.shape
    assert N == M, "a square matrix is required"

    # find set of nodes which have zero probability
    keep_idxs = np.nonzero(np.sum(prob, 1) > 0)[0]

    # convert probabilities to distances
    dist = -np.log(prob)

    # complete distances
    dist_func = graph_shortest_path(dist)

    # slice distance function 
    dist_func = dist_func[keep_idxs,:][:,keep_idxs]
    
    return dist_func, keep_idxs


def get_map(prob, perplexity=10.0, n_iter=1000, learning_rate=50.0):
    """
    Create a 2-D t-SNE map from the given probability matrix.
    Returns dict {idx: (x,y)}.
    """
    
    dist_func, keep_idxs = prob_to_dist_func(prob)

    tsne = TSNE(
        metric="precomputed",
        perplexity=perplexity,
        n_iter=n_iter,
        learning_rate=learning_rate
    )
    y = tsne.fit_transform(dist_func)
    
    return {k: y[i,:] for i,k in enumerate(keep_idxs)}


# ===================================================================
# Experimental code (not used yet)
# ===================================================================

class UpperBound:

    def __init__(self, k):
        self._k = k
        self._q = []

    @property
    def threshold(self):
        return -self._q[0]
        
    def visit(self, value):
        k = self._k
        if len(self._q) == k:
            threshold = -self._q[0]
            if value >= threshold: return
            heappush(q, -value)
            heappop(q)
        else:
            heappush(q, -value)


def graph_nn(start_node):
    """Yield nearest-neighbors from graph in order."""

    pq = []
    for p, d in neighbors(start_node):
        heappush(pq, (d, p))
        
    visited = set([start_node])
    while pq and count < k:
        dist, node = heappop(pq)
        yield (node, dist)
        visited.add(node)
        for p, d in neighbors(node):
            if p not in visited:
                newd = dist + d
                heappush(pq, (newd, p))


def graph_knn(start_node, k):
    pq = []
    for p, d in neighbors(start_node):
        heappush(pq, (d, p))
    
    # by tracking the upper bound, we avoid potentially placing lots of unnecessary items in the priority queue
    visited = set([start_node])
    ub = UpperBound(k)
    while pq and len(visited) < k + 1:
        dist, node = heappop(pq)
        yield (node, dist)
        visited.add(node)
        for p, d in neighbors(node):
            newd = dist + d
            if newd < ub.threshold:
                heappush(pq, (newd, p))
                ub.visit(newd)



