import networkx as nx
import numpy as np


def nfftriads(G):
    """Counts the number of feed-forward triads in directed graph `G`.

    A feedforward triad is an ordered triplet `(i,j,k)` of nodes such that
    `(i,j)`, `(i,k)`, and `(j,k)` are all edges in `G`"""

    A = nx.to_numpy_matrix(G)
    return sum(np.multiply(np.matmul(A, A), A))


def nreciprocalpairs(G):
    """Counts the number of reciprocal pairs in directed graph `G`.
    A reciprocal pair is an unordered pair `{i,j}` of nodes such that `(i,j)` and `(j,i)` are edges of `G`."""

    A = nx.to_numpy_matrix(G)
    return np.trace(np.matmul(A, A)) / 2.0


def n3cycles(G):
    """Counts the number of directed 3-cycles in directed graph `G`.

    A directed 3-cycle is an ordered triplet `(i,j,k)` of nodes such that
    `(i,j)`, `(j,k)`, and `(k,i)` are all edges in `G`."""

    A = nx.to_numpy_matrix(G)
    return np.trace(np.matmul(np.matmul(A, A), A))
