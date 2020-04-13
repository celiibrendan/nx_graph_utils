"""
A pure-python implementation of exponential random graph models (ERGMs). Adapted from https://github.com/jcatw/ergm

Throughout this package, graphs are represented by their adjacency matrices, which are simply numpy arrays.

Classes:
    ergmpy
"""
import numpy as np

from ergmpy import util


class ergm:
    def __init__(self, stats, coeffs=None, directed=False):
        """
        Construct an ergmpy with a given family of sufficient statistics.

        Args:

            stats : a list of functions which compute the sufficient statistics of the family. Each function should
                    take as input 2d numpy array representing the adjacency matrix and return a numeric value.

            coefs : a list of the natural parameters which determine the distribution. Default is all zeros.

            directed : whether the graphs from this family are directed or not. Default False.
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
        return np.exp(np.dot(util.pam(self.stats, A), self.coeffs))

    def sample_binary(self, n_nodes, n_samples=1, A_0=None, burn_in=None, block_size=1, n_steps=None, order="columns",
                      dtype=int, verbose=0):
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

            dtype : the data type of the adjacency matrix. Default is int, for binary adjacency matrices.

            order : The order the entries in the adjacency matrix are filled in. Default "columns"

            verbose : how much output to print. Useful for debugging purposes. Valid values are 0 (default,
                no output), 1 (a little output), 2 (a little more

        Returns:
            A 3d numpy array with shape (n_nodes, n_nodes, n_samples).
        """
        if A_0 is None:
            A = np.zeros((n_nodes, n_nodes))
        else:
            A = A_0  # current adjacency matrix, will be updated in place.
        if burn_in is None:
            burn_in = 10 * n_nodes
        if n_steps is None:
            n_steps = 10 * n_nodes
        if verbose > 0:
            if self.directed:
                type_str = "directed"
            else:
                type_str = "undirected"
            print("Sampling a {} node {} graph using Gibbs sampler".format(n_nodes, type_str))
            print("  burn-in:            {}".format(burn_in))
            print("  inter-sample steps: {}".format(n_steps))
            print("  block size:         {}".format(block_size))

        n_edges = n_nodes * (n_nodes - 1) // (1 + (not self.directed))

        samples = np.empty((n_nodes, n_nodes, n_samples), dtype=dtype)

        total_updates = (burn_in + n_steps * n_samples) * block_size
        if verbose > 1:
            print("Preparing sequence of edges to test")
        # idx_sequence = np.random.choice(n_edges, size=(total_updates,))
        # edge_indices = np.stack([np.random.choice(n_edges, block_size, replace=False) for k in range(burn_in + n_steps * n_samples)])
        # # edges = map(lambda k: util.index_to_edge(k, n_nodes, self.directed, order), idx_sequence)
        # if verbose > 1:
        #     # print("First few edges to be sampled: {}".format(idx_sequence[:10]))
        #     print("First few edges to be sampled: {}".format(edge_indices[:10]))
        #     # if verbose > 2:
        #     #     print("Histogram of choices:")
        #     #     print(np.histogram(idx_sequence, bins=(n_nodes * (n_nodes-1))))

        for bk in range(burn_in + n_steps * n_samples):
            # S = util.index_to_edge(idx_sequence[(bk * block_size):((bk + 1) * block_size)], n_nodes,
            #                        directed=self.directed)
            # S = util.index_to_edge(edge_indices[bk,:], n_nodes, directed=self.directed)

            # seems like i'm running out of memory, possibly? When pre-allocating the whole sequence.
            S = util.index_to_edge(np.random.choice(n_edges, block_size, replace=False), n_nodes, directed=self.directed)
            if verbose > 2 and (bk == 0 or (0 < bk and (bk - burn_in) % n_steps == 0)):
                print("  Step {:3d}, sampling edges {}".format(bk, S))

            p_config = np.empty(2 ** block_size)
            # for i in range(2 ** block_size):  # looping over all possible configurations of those edges
            #     li = util.binlist(i)  # this is the specific configuration, should be same length as S
            #     for S_i, A_i in zip(S, li):
            #         A[S_i[0], S_i[1]] = A_i
            #     p_config[i] = np.exp(np.dot(util.pam(self.stats, A), self.coeffs))  # weight of this configuration
            configs = util.binary_digits(np.arange(2 ** block_size), block_size)
            for i in range(2 ** block_size):
                A[tuple(S)] = configs[i, :]
                # p_config[i] = self.weight(A)
                p_config[i] = np.exp(np.dot(util.pam(self.stats, A), self.coeffs))  # weight of this configuration
                if verbose > 2 and (bk == 0 or (0 < bk and bk - burn_in) % n_steps == 0):
                    print("    Configuration {} has  weight {}".format(configs[i, :], p_config[i]))

            p_config = p_config / np.sum(p_config)
            config = np.random.choice(range(2 ** block_size), p=p_config)
            # li = util.binlist(config)
            # for S_i, A_i in zip(S, li):
            #     A[S_i[0], S_i[1]] = A_i
            A[tuple(S)] = configs[config, :]
            if bk >= burn_in and (bk - burn_in) % n_steps == 0:
                samples[:, :, (bk - burn_in) // n_steps] = A[:, :]

        return samples
