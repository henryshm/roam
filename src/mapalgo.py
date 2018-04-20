# Roam - Copyright 2018 David J. C. Beach; see LICENSE in project root

"""
# Attribution

This code is originally based on code found in the article
[In Raw Numpy: t-SNE](https://nlml.github.io/in-raw-numpy/in-raw-numpy-t-sne/)
by Liam Schoneveld, September 18, 2017.

It has been heavily modified to support abstracted parameters, prior maps,
and other features.

"""

from collections import namedtuple
from collections import defaultdict
from sklearn import decomposition
import itertools
import functools

import numpy as np
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans


#: Optimization Parameters for gradient descent
OptParams = namedtuple("OptParams", [
    "num_iters",        # maximum number of iterations
    "learning_rate",    # learning rate in gradient descent
    "momentum",         # momentum in gradient descent
    "seed",             # random seed to use in initialization (or None)
])


DEFAULT_OPT_PARAMS = OptParams(
    num_iters=1000,
    learning_rate=10.0,
    momentum=0.9,
    seed=12345,
)


def neg_squared_euc_dists(X):
    """
    Compute matrix containing negative squared euclidean
    distance for all pairs of points in input matrix X

    # Arguments:
        X: matrix of size NxD
    # Returns:
        NxN matrix D, with entry D_ij = negative squared
        euclidean distance between rows X_i and X_j
    """
    # Math? See https://stackoverflow.com/questions/37009647
    sum_X = np.sum(np.square(X), 1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    return -D


def softmax(X, diag_zero=True):
    """Take softmax of each row of matrix X."""

    # Subtract max for numerical stability
    e_x = np.exp(X - np.max(X, axis=1).reshape([-1, 1]))

    # We usually want diagonal probabilities to be 0.
    if diag_zero:
        np.fill_diagonal(e_x, 0.)

    # Add a tiny constant for stability of log we take later
    e_x = e_x + 1e-8  # numerical stability

    return e_x / e_x.sum(axis=1).reshape([-1, 1])


def calc_prob_matrix(distances, sigmas):
    """Convert a distances matrix to a matrix of probabilities."""
    two_sig_sq = 2. * np.square(sigmas.reshape((-1, 1)))
    return softmax(distances / two_sig_sq)


def binary_search(eval_fn, target, tol=1e-10, max_iter=10000, 
                  lower=1e-20, upper=1000.):
    """
    Perform a binary search over input values to eval_fn.
    
    # Arguments
        eval_fn: Function that we are optimising over.
        target: Target value we want the function to output.
        tol: Float, once our guess is this close to target, stop.
        max_iter: Integer, maximum num. iterations to search for.
        lower: Float, lower bound of search range.
        upper: Float, upper bound of search range.
    # Returns:
        Float, best input value to function found during search.
    """
    for i in range(max_iter):
        guess = (lower + upper) / 2.
        val = eval_fn(guess)
        if val > target:
            upper = guess
        else:
            lower = guess
        if np.abs(val - target) <= tol:
            break
    return guess


def calc_perplexity(prob_matrix):
    """
    Calculate the perplexity of each row 
    of a matrix of probabilities.
    """
    entropy = -np.sum(prob_matrix * np.log2(prob_matrix), 1)
    perplexity = 2 ** entropy
    return perplexity


def perplexity(distances, sigmas):
    """Wrapper function for quick calculation of 
    perplexity over a distance matrix."""
    return calc_perplexity(calc_prob_matrix(distances, sigmas))


def find_optimal_sigmas(distances, target_perplexity):
    """
    For each row of distances matrix, find sigma that results
    in target perplexity for that role.
    """
    sigmas = [] 
    # For each row of the matrix (each point in our dataset)
    for i in range(distances.shape[0]):
        # Make fn that returns perplexity of this row given sigma
        eval_fn = lambda sigma: \
            perplexity(distances[i:i+1, :], np.array(sigma))
        # Binary search over sigmas to achieve target perplexity
        correct_sigma = binary_search(eval_fn, target_perplexity)
        # Append the resulting sigma to our output array
        sigmas.append(correct_sigma)
    return np.array(sigmas)


def q_joint(Y):
    """Given low-dimensional representations Y, compute
    matrix of joint probabilities with entries q_ij."""
    # Get the distances from every point to every other
    distances = neg_squared_euc_dists(Y)
    # Take the elementwise exponent
    exp_distances = np.exp(distances)
    # Fill diagonal with zeroes so q_ii = 0
    np.fill_diagonal(exp_distances, 0.)
    # Divide by the sum of the entire exponentiated matrix
    return exp_distances / np.sum(exp_distances), None


def p_conditional_to_joint(P):
    """
    Given conditiohnal probabilities matrix P, return
    approximation of joint distribution probabilities.
    """
    return (P + P.T) / (2. * P.shape[0])


def p_joint(X, target_perplexity):
    """
    Given a data matrix X, gives joint probabilities matrix.

    # Arguments
        X: Input data matrix.
    # Returns:
        P: Matrix with entries p_ij = joint probabilities.
    """
    # Get the negative euclidian distances matrix for our data
    distances = neg_squared_euc_dists(X)
    # Find optimal sigma for each row of this distances matrix
    sigmas = find_optimal_sigmas(distances, target_perplexity)
    # Calculate the probabilities based on these optimal sigmas
    p_conditional = calc_prob_matrix(distances, sigmas)
    # Go from conditional to joint probabilities matrix
    P = p_conditional_to_joint(p_conditional)
    return P


def symmetric_sne_grad(P, Q, Y, _):
    """Estimate the gradient of the cost with respect to Y."""
    pq_diff = P - Q  # NxN matrix
    pq_expanded = np.expand_dims(pq_diff, 2)  #NxNx1
    y_diffs = np.expand_dims(Y, 1) - np.expand_dims(Y, 0)  #NxNx2
    grad = 4. * (pq_expanded * y_diffs).sum(1)  #Nx2
    return grad


def estimate_sne(P, q_fn, grad_fn, params):
    """
    Estimates a SNE model.

    # Arguments
        P: Matrix of joint probabilities.
        q_fn: Function that takes Y and gives Q prob matrix.
        grad_fn: Function to compute gradient cost, given (P, Q, Y, inv_distances)
    # Returns:
        Y: Matrix, low-dimensional representation of X.
    """

    N, M = P.shape
    assert N == M, "P must be a square matrix"
    assert isinstance(params, OptParams), "an OptParams instance is required for params"
    
    # Initialise our 2D representation
    rng = np.random.RandomState(seed=params.seed)
    Y = rng.normal(0., 0.0001, [N, 2])

    # Initialise past values (used for momentum)
    if params.momentum:
        Y_m2 = Y.copy()
        Y_m1 = Y.copy()

    # Start gradient descent loop
    for i in range(params.num_iters):

        # Get Q and distances (distances only used for t-SNE)
        Q, inv_distances = q_fn(Y)
        # Estimate gradients with respect to Y
        grads = grad_fn(P, Q, Y, inv_distances)

        # Update Y
        Y = Y - params.learning_rate * grads
        if params.momentum:  # Add momentum
            Y += params.momentum * (Y_m1 - Y_m2)
            # Update previous Y's for momentum
            Y_m2 = Y_m1.copy()
            Y_m1 = Y.copy()

    return Y


def q_tsne(Y):
    """t-SNE: Given low-dimensional representations Y, compute
    matrix of joint probabilities with entries q_ij."""
    distances = neg_squared_euc_dists(Y)
    inv_distances = np.power(1. - distances, -1)
    np.fill_diagonal(inv_distances, 0.)
    return inv_distances / np.sum(inv_distances), inv_distances


def tsne_grad(P, Q, Y, inv_distances):
    """Estimate the gradient of t-SNE cost with respect to Y."""
    pq_diff = P - Q
    pq_expanded = np.expand_dims(pq_diff, 2)
    y_diffs = np.expand_dims(Y, 1) - np.expand_dims(Y, 0)

    # Expand our inv_distances matrix so can multiply by y_diffs
    distances_expanded = np.expand_dims(inv_distances, 2)

    # Multiply this by inverse distances matrix
    y_diffs_wt = y_diffs * distances_expanded

    # Multiply then sum over j's
    grad = 4. * (pq_expanded * y_diffs_wt).sum(1)
    return grad


def tsne_grad_prior(P, Q, Y, inv_distances, Y0, Y0mask, gamma=0.01):
    """
    Modified t-SNE gradient which includes a squared distance penalty term
    when points are moved away from some location in a prior map.
    """
    grad = tsne_grad(P, Q, Y, inv_distances)
    N, _ = Y.shape
    y_diffs = Y - Y0
    y_diffs[~Y0mask,:] = 0.0
    y_diffs *= (gamma / N)
    grad += y_diffs
    return grad


def estimate_tsne(P, params):
    """
    Estimates a t-SNE model.

    # Arguments
        P: Matrix of joint probabilities.
        params: Optimization parameters.
    # Returns:
        Y: Matrix, low-dimensional representation of X.
    """
    N, _ = P.shape
    assert P.shape == (N,N), "P must be square"
    y = estimate_sne(
        P,
        q_fn=q_tsne,
        grad_fn=tsne_grad,
        params=params,
    )
    return y


def estimate_tsne_prior(
        P,
        Y0,
        Y0mask,
        params,
        gamma=0.01,
    ):
    """Prior t-SNE model."""
    N, _ = P.shape
    assert P.shape == (N,N), "P must be square"
    assert Y0.shape == (N,2), "Y0 must have shape (N,2)"
    assert Y0mask.shape == (N,), "Y0mask should be 1-D of length N"

    grad = functools.partial(tsne_grad_prior, Y0=Y0, Y0mask=Y0mask, gamma=gamma)
    y = estimate_sne(
        P,
        q_fn=q_tsne,
        grad_fn=grad,
        params=params,
    )
    return y


def renorm_probs(probs):
    """
    Renormalize a probability matrix, returning a matrix in which
    each element p_ij satisfies:
    
       * 0 <= p_ij <= 1
       * p_ii = 0
       * p_ij = p_ji
       * sum(p_ij) = 1
    
    """

    # clip any negative values
    p = np.clip(probs, 0.0, np.inf)

    # diagonal must be zero
    np.fill_diagonal(p, 0.0)

    # ensure symmetry
    p += p.T

    # ensure unit sum
    p /= np.sum(p)

    return p


Map2D = namedtuple("Map2D", ["name", "labels", "prob", "y", "prior", "params"])


def create_map(name, prob,
        labels=None,
        prior=None,
        gamma=0.01,
        params=DEFAULT_OPT_PARAMS,
    ):
    """
    Create a map with given name and probability matrix, P.
    An optional "prior" map may be provided.

    Returns a Map2D object with y values determined by estimating
    t-SNE (x, y) positions to estimate P.
    """
    prob = renorm_probs(prob)
    N = prob.shape[0]
    if labels is None:
        labels = np.arange(N)
    else:
        assert len(labels) == N, "labels must match dimension of prob matrix"
        labels = np.asarray(labels)

    # reduce dimension of problem to only those indices for which we
    # have nonzero probabilities
    keep = np.sum(prob, 1) > 1e-6
    if not np.any(keep):
        return Map2D(name, np.zeros(0), np.zeros((0,0)), np.zeros((0,2)), prior, params)
    if not np.all(keep):
        labels = labels[keep]
        prob = prob[np.ix_(keep, keep)]
        N = np.sum(keep)

    if prior is None:
        y = estimate_tsne(prob, params=params)
    else:
        y0data = np.zeros((N,2))
        y0mask = np.zeros(N, dtype=bool)
        lab2idx = {l: i for i, l in enumerate(labels)}
        for j, l in enumerate(prior.labels):
            i = lab2idx.get(l)
            if i is not None:
                y0data[i,:] = prior.y[j,:]
                y0mask[i] = True

        y = estimate_tsne_prior(
            prob,
            Y0=y0data,
            Y0mask=y0mask,
            gamma=gamma,
            params=params,
        )
    return Map2D(name, labels, prob, y, prior, params)


def translate_labels(src, dest):
    """
    Create a mapping from the labels in src to the labels in dest.
    Returns arrays (remap, mask) such that src[remap] = dest[mask],
    where remap is a permutation/selection of labels in src,
    and mask is a boolean mask of labels in dest.
    """
    labmap = {l:i for i, l in enumerate(src)}
    remap = []
    for l in dest:
        idx = labmap.get(l)
        if idx is not None:
            remap.append(idx)
    remap = np.array(remap)
    mask = np.array([l in labmap for l in dest])
    return remap, mask


def create_map_from_data(name, data,
        labels=None,
        perplexity=20.0,
        prior=None,
        gamma=0.01,
        params=DEFAULT_OPT_PARAMS,
    ):
    """
    Creates a t-SNE map from a data matrix with the desired perplexity.
    """
    if labels is None:
        labels = np.arange(len(data))
    prob = p_joint(data, target_perplexity=perplexity)

    #t-SNE version
    # return create_map(name, prob,
    #     labels=labels,
    #     prior=prior,
    #     gamma=gamma,
    #     params=params,
    # )

    #PCA code
    print("N = {}".format(len(data)))
    pca = decomposition.PCA(n_components=26)
    pca.fit(data)
    y = pca.transform(data)
    print(y)

    return Map2D(name, labels, prob, y[:, 1:3], prior, params)


def create_blended_map(
        name, maps, weights, preserve=0, prior=0, 
        gamma=0.01,
        params=DEFAULT_OPT_PARAMS,
    ):
    """
    Create a blended map by combining two or more other maps.

    preserve: preserve labels from the first k maps only
    prior: use map at this index as the prior (or another map instance, or None)
    """

    k = len(maps)
    assert k >= 2, "at least two maps are required"
    assert len(weights) == k, "must have corresponding number of maps and weights"

    # create a set of used labels from all maps to be blended
    # (while attempting to preserve original order)
    # also build the mapping from label to index
    all_labels = []
    lmap = {}
    for s, m in enumerate(maps):
        if s > preserve: break
        for l in m.labels:
            if l not in lmap:
                i = len(all_labels)
                all_labels.append(l)
                lmap[l] = i
    all_labels = np.array(all_labels)

    # normalize the weights
    weights = np.array(weights)
    np.clip(weights, 0.0, np.inf)
    weights /= np.sum(weights)

    # compute the blended probability matrix
    N = len(all_labels)
    Pmix = np.zeros((N,N))
    for i, m in enumerate(maps):
        # index translation selects corresponding position in all_labels
        # for each label in map
        tr = np.array([lmap.get(l, -1) for l in m.labels])
        rmask = (tr >= 0)
        lindex = tr[rmask]
        Pmix[np.ix_(lindex, lindex)] += weights[i] * m.prob[np.ix_(rmask, rmask)]

    # select appropriate prior
    if isinstance(prior, int):
        prior = maps[prior]

    return create_map(name, Pmix,
        prior=prior,
        gamma=gamma,
        params=params,
    )


def weights_to_condprob(weights):
    """
    Given a square matrix of weights or tallies, convert to a matrix of
    conditional probabilities.  Each entry in the resulting matrix,
    p_{ij} = p_{j|i}, the probability of node j given node i.  Any diagonal
    entries of the weights matrix are ignored when computing the conditional
    probabilities, since the probability of a node given itself is
    (trivially) p_{i|i} = 1.
    """
    weights = np.array(weights, dtype=float)
    N, M = weights.shape
    assert N == M, "a square matrix is required"

    # exclude any diagonal weights by setting them to zero
    np.fill_diagonal(weights, 0.0)

    # compute sum of weights for each node
    wsum = np.sum(weights, 1)
    wtot = np.sum(weights)

    if wtot < 1e-6:
        return np.zeros((N,N))

    # avoid divide-by-zero for nodes that have no weight
    zeroidx = ((wsum/wtot) < 1e-6)
    wsum[zeroidx] = 1.0

    # compute conditional probabilities
    condprob = (weights / wsum.reshape([-1, 1]))

    condprob[:,zeroidx] = 0.0

    return condprob


def propagate_with_decay(P, n=3, gamma=0.1):
    """
    Given a conditional probability matrix P, propagate those probabilities
    using P as a probabilistic state transition matrix, by exponentiating P.
    The result is a sum of the exponents of P, up to power n, where
    an exponentially decaying coefficient gamma is applied to each term.

    [(P*gamma)^1 + (P*gamma)^2 + ... + (P*gamma^n)] / [gamma^1 + gamma^2 + ... + gamma^n]

    Note that if the input is a valid conditional probability matrix, then the
    result will also be a valid conditional probability matrix, with the probabilities
    "spread" across more of the reachable nodes from each state.

    """

    N = P.shape[0]
    assert P.shape == (N,N), "P must be a square matrix of conditional probabilities"
    assert 0 <= n, "n should be a nonnegative integer"
    assert 0.0 <= gamma <= 1, "gamma should be in the range [0,1]"

    if n == 0:
        return np.eye(N)

    Pgamma = P * gamma
    A = Pgamma
    numer = Pgamma.copy()
    denom = gamma

    for i in range(2, n+1):
        A = np.dot(A, Pgamma)
        numer += A
        denom += pow(gamma, i)

    numer /= denom
    return numer


def zscore(X):
    """
    Convert columns of X to z-scored data.

    Columns with a near-zero standard deviation are mean-centered only.
    """
    N, D = X.shape
    means = np.tile(np.mean(X, 0), (N,1))
    stds = np.tile(np.std(X, 0), (N,1))
    stds[stds < 1e-12] = 1.0
    Z = (X - means) / stds
    return Z


def map_mean_sse(map1, map2):
    """
    Return the Mean Sum of Squared Error (Mean SSE), based on the
    movement of points between maps.

    A zero value indicates identical maps (for all points with common label),
    and higher values indicate increasing dissimilarity.  This value is
    only meaningful if the two maps are based on some common prior
    (either directly or though some more distant ancestral relationship).
    """
    remap, mask = translate_labels(map2.labels, map1.labels)

    y1 = map1.y[mask]
    y2 = map2.y[remap]

    N, _ = y1.shape

    delta_y = y2 - y1
    terms = delta_y[:,0]**2
    terms += delta_y[:,1]**2
    sse = np.sum(terms)
    mean_sse = sse / N

    return mean_sse


def stabilize_assignments(old_assign, new_assign):
    """
    Returns the mapping from old assignment labels to new assignment labels
    which is most parsimonious so as to avoid having points move between groups.
    In other words:

    mapping[old_assign] ~~ new_assign

    has a minimum of changed labels as compared with any other label
    permutation mapping.
    """
    old_assign = np.asarray(old_assign)
    new_assign = np.asarray(new_assign)
    assert len(old_assign) == len(new_assign)
    k = np.max(old_assign) + 1
    N = len(old_assign)

    # find the best assignment of old_assign on to new_assign,
    # based on the total number of common points
    best_score = -1.0
    best_assignment = None
    for candidate in itertools.permutations(list(range(k))):
        score = 0.0
        for i, j in zip(range(k), candidate):
            score += np.sum((old_assign == i) & (new_assign == j))
        if score > best_score:
            best_score = score
            best_assignment = candidate

    return np.array(best_assignment)


# Dark2 Scheme (qualitative) with 8 categories:
# http://colorbrewer2.org/#type=qualitative&scheme=Dark2&n=8
_CATEGORY_COLORS = (
    "#1b9e77",
    "#d95f02",
    "#7570b3",
    "#e7298a",
    "#66a61e",
    "#e6ab02",
    "#a6761d",
    "#666666",
)


def show_map_matrix(
        maps,
        keys=None,
        k=6,
        alpha=0.6,
        size=5,
        score=True,
        colormap=_CATEGORY_COLORS,
        unmatched_color="#C0C0C0"
    ):
    """
    Create a matrix of scatterplots (but technically not a SPLOM),
    where each column corresponds to a given map, but each row uses kmeans class
    labels derived from the map in that row.

    This enables fast visual comparison of the course-level organization in each map.
    By scanning across a row, one can see the degree to which clusters in one map
    are preserved in other maps.  This requires the maps to share some common labels,
    and works best when maps are organized based on some related prior map.

    # Arguments
        maps: List of maps to be compared
        k: Int, Number of clusters to create from each map
        alpha: Float, alpha level to use when plotting
        size: Int, size of single frame (given as side length)
        score: Include Mean-SSE score between plots
        colormap: List of colors to use for category labels
        unmatched_color: Color to use for unmatched points

    """

    if keys is None:
        keys = maps
    else:
        for x in keys:
            for y in maps:
                if x is y: break
            else:
                assert False, "keys must be subset of maps"

    getcolor = lambda c: colormap[c % len(colormap)]

    nmaps = len(maps)
    nkeys = len(keys)
    
    fig, ax = plt.subplots(nkeys, nmaps, figsize=(nmaps*size,nkeys*size), squeeze=False)
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, hspace=0.1, wspace=0.1)

    assign = [
        KMeans(n_clusters=k).fit(m.y).labels_
        for m in keys
    ]
    # stabilize the assigned labels
    for i in range(1,nkeys):
        # create mapping from prev labels to current labels
        remap, mask = translate_labels(keys[i-1].labels, keys[i].labels)
        assignmap = stabilize_assignments(assign[i][mask], assign[i-1][remap])
        assign[i] = assignmap[assign[i]]

    for row in range(nkeys):
        rowlabels = assign[row]
        rowids = keys[row].labels
        cmap = {rowids[i]: getcolor(rowlabel) for i, rowlabel in enumerate(rowlabels)}
        for col in range(nmaps):
            plot = ax[row,col]
            data = maps[col].y
            colors = [cmap.get(id, unmatched_color) for id in maps[col].labels]
            title = maps[col].name
            if score:
                s = map_mean_sse(keys[row], maps[col])
                title += " - %5.2f" % s
            plot.scatter(data[:,0], data[:,1], alpha=alpha, color=colors)
            plot.set_title(title)
            plot.axis("equal")
            for axis in [plot.xaxis, plot.yaxis]:
                axis.set_major_formatter(plt.NullFormatter())
            framewidth = 3 if keys[row] is maps[col] else 0.5
            for feat in ["left", "top", "right", "bottom"]:
                plot.spines[feat].set_color("gray")
                plot.spines[feat].set_linewidth(framewidth)


