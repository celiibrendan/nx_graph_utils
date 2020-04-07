"""
A pure-python implementation of exponential random graph models (ERGMs). Adapted from https://github.com/jcatw/ergm

Throughout this package, graphs are represented by their adjacency matrices, which are simply numpy arrays.

Classes:
    ergmpy
"""
import numpy as np

from ergmpy import util


class ergm:
    def __init__(self, stats, coeffs=None, directed=True):
        """
        Construct an ergmpy with a given family of sufficient statistics.

        Args:

            stats : a list of functions which compute the sufficient statistics of the family. Each function should
                    take as input 2d numpy array representing the adjacency matrix and return a numeric value.

            coefs : a list of the natural parameters which determine the distribution. Default is all zeros.

            directed : whether the graphs from this family are directed or not. Default true.
        """
        self.stats = stats
        if coeffs is None:
            self.coeffs = [0.] * len(stats)
        else:
            assert len(coeffs) == len(stats)
            self.coeffs = coeffs
        self.directed = directed

    def weight(self, A):
        """
        Compute the unnormalized probability of a graph under this distribution.

        Args:

            A : A 2d numpy array representing an adjacency matrix.
        """
        return np.exp(np.sum(util.pam(self.stats, A)))

    def sample_binary(self, n_nodes, n_samples=1, A_0=None, burn_in=None, block_size=1, n_steps=None, order="columns",
                      directed=False, dtype=int):
        """
        Return samples from this distribution using Gibbs sampling. Assumes distribution is over simple, "undecorated"
        graphs (i.e. no  edge weights, no node properties, no self-loops).

        Args:
            n_nodes : the number of nodes in the graph

            n_samples : number of samples to return. Default 1.

            A_0 : starting graph. Default value of None results in empty graph (np.zeros) being used.

            burn_in : number of initial, discard steps to take before returning samples. Default is n_nodes ** 2

            block_size : the number of edges to try flipping at once. Default is 1.

            n_steps : the number of markov chain steps to take between samples. Default is n_nodes ** 2

            directed : the

            dtype : the data type of the adjacency matrix. Default is int, for binary adjacency matrices.

            order : The order the entries in the adjacency matrix are filled in. Default "columns"

        Returns:
            A 3d numpy array with shape (n_nodes, n_nodes, n_samples).
        """
        if A_0 is None:
            A = np.zeros((n_nodes, n_nodes))
        else:
            A = A_0  # current adjacency matrix, will be updated in place.
        if burn_in is None:
            burn_in = n_nodes ** 2
        if n_steps is None:
            n_steps = n_nodes ** 2

        samples = np.zeros((n_nodes, n_nodes, n_samples), dtype=dtype)

        total_updates = (burn_in + n_steps * n_samples) * block_size
        idx_sequence = np.random.choice(range(n_nodes * (n_nodes - 1) // (1 + (not self.directed))),
                                        size=(total_updates,))
        # edges = map(lambda k: util.index_to_edge(k, n_nodes, self.directed, order), idx_sequence)

        # sample_steps = np.arange(burn_in * block_size, total_updates, n_steps * block_size) + n_steps * block_size
        for bk in range(burn_in + n_steps * n_samples):
            S = [util.index_to_edge(idx, n_nodes, directed) for idx in
                 idx_sequence[(bk * block_size):((bk + 1) * block_size)]]
            # S is the list of edges (i.e. ordered pairs)
            p_e = np.zeros(2 ** block_size)
            for i in range(2 ** block_size):  # looping over all possible configurations of those edges
                li = util.binlist(i)  # this is the specific configuration, should be same length as S
                # for idx, edge in enumerate(li):
                #     S_i = util.index_to_edge(S[idx], n_nodes, directed=directed)
                #     # A[S[idx][0], S[idx][1]] = edge
                for S_i, A_i in zip(S, li):
                    A[S_i[0], S_i[1]] = A_i
                p_e[i] = np.exp(np.dot(util.pam(self.stats, A), self.coeffs))  # weight of this configuration
            p_e = [p / sum(p_e) for p in p_e]
            config = np.random.choice(range(2 ** block_size), p=p_e)
            li = util.binlist(config)
            # for idx, edge in enumerate(li):
            #     A[S[idx][0], S[idx][1]] = edge
            for S_i, A_i in zip(S, li):
                A[S_i[0], S_i[1]] = A_i
            if bk >= burn_in and (bk - burn_in) % n_steps == 0:
                samples[:, :, (bk - burn_in) // n_steps] = A[:, :]

        return samples
