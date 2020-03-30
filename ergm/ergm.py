"""
A pure-python implementation of exponential random graph models (ERGMs). Adapted from https://github.com/jcatw/ergm

Throughout this package, graphs are represented by their adjacency matrices, which are simply numpy arrays.

Classes:
    ergm
"""
import numpy as np

from ergm import util


class ergm:
    def __init__(self, stats, coeffs=None, directed=True):
        """
        Construct an ergm with a given family of sufficient statistics.

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


