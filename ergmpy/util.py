"""
Utility functions for the ergmpy package
"""
import numpy as np


def pam(fs, x):
    """
    Apply a sequence of functions to x. Like map, but backwards, hence the name.
    This is equivalent to `list(map(lambda f: f(x), fs))`.

    Args:

        fs : a sequence of functions

        x : a value to which each f in fs is applied

    Returns:
        The list [f(x) for f in fs].
    """
    return [f(x) for f in fs]


def binlist(n):
    """Returns integer `n` as a list of binary digits (no padding)."""
    return [int(x) for x in bin(n)[2:]]


def binary_digits(n, d):  # numpy-optimized
    """Returns an n x d array of the binary digits of each entry of array n

    Parameters:
        n : array_like
            Integer values to be represented as binary digits

        d : the number of digits; zero padding and/or truncation if necessary

    Returns:
        digits : an n x d binary array; each row is the digits of the corresponding entry of n. Least significant bit has index 0.
    """
    return ((n[:, None] & (1 << np.arange(d))) > 0).astype(int)
# def edge_index(e0, e1, n, directed=False, order="columns"):
#     """Returns the index of the edge `(e0,e1)` in an `n`-node graph."""
#     if order == "columns":
#         return e0 * n + e1 + (e1 > e0)
#     else:
#         return e1 * n + e0 + (e0 > e1)


def index_to_edge_old(idx, n, directed=True, order="columns"):
    """Returns the ordered pair `(e0,e1)` for the edge which has linear index `idx`. This is essentially the linear
    index of an entry in a matrix, except shifts are included so the diagonal entries don't get indexed.

    Args:

        idx : an integer between 0 and n*(n-1) (inclusive) for directed graphs, or 0 and n*(n-1)/2 for undirected.

        n : the number of nodes in the graph

        directed : whether to find the index for a directed (all off-diagonal entries used) or undirected
                   (upper triangle only). Default: true

        order : Whether matrix entries are indexed in column order or row order. Default columns, so 0 maps to (1,0),
                and so on down the 0th column before moving to the 1th column. Options are "columns" (default)
                or "rows".

    Returns:
        A tuple of integers, the indices in the adjacency matrix.
    """
    if directed:
        e1 = idx // (n - 1)
        e0 = idx % (n - 1) + (idx % (n - 1) >= e1)
        if order == "columns":
            return (e0, e1)
        else:
            return (e1, e0)
    else:
        e1 = int(np.ceil(triangular_root(idx + 1)))
        e0 = idx - (e1 - 1) * e1 // 2
        if order == "columns":
            return (e0, e1)
        else:
            return (e1, e0)


def index_to_edge(idx, n, directed=True, order="C"):
    """Converts linear indices to edge tuples. Behaves like numpy.unravel_index, with some slight modifications:
    1. It does not return tuples of the form (i,i) and 2. If `directed=False`, the tuples are always in increasing order.

    Parameters:
        idx : array_like
            An integer array whose elements are indices into the edges of the graph; each entry must be nonnegative and less than `n * (n-1).

        n : the number of vertices in the graph

        directed : whether the graph is directed. Default True.

        order : 'C' for C-style (i.e. rows filled first) or 'F' for Fortran-style (i.e. columns filled first) indexing.

    Returns:
        edges : a 2-tuple of arrays, each the same size as `idx`.
    """
    I, J = np.unravel_index(idx, (n-1,n), 'F')
    I[I >= J] += 1
    if directed:
        I, J = np.min(np.vstack((I, J)), axis=0), np.max(np.vstack((I, J)), axis=0)
    if order == "C":
        return I, J
    else:
        return J, I


def triangular_root(x):
    """Returns the triangular root of x. If this returns an integer, x is a triangular number; otherwise, it lies between two triangular numbers.

    See https://en.wikipedia.org/wiki/Triangular_number"""
    return (np.sqrt(8 * x + 1) - 1) / 2


def triangular_number(n):
    """Returns the `n`th triangular number `n * (n + 1) // 2`"""
    return n * (n + 1) // 2
