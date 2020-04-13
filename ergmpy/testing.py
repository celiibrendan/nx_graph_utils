"""Testing functionality of the ergmpy package"""
import math

import numpy as np
from ergmpy import ergm
import networkx as nx
import time
from scipy.special import binom

print("Testing ergmpy")

directed_test = True

print("  Directed Graphs: {}".format(directed_test))

p = 0.34  # parameter for ER random graphs

n_nodes = 6
n_samples = 10000
# seed = 1717

print("Using Networkx to sample Erdos-Renyi random graphs with edge probability p = {}".format(p))
print("Producing {} samples with {} nodes".format(n_samples, n_nodes))
# print("Using seed {}".format(seed))

nx_ER_start = time.time()
nx_ER_list = [nx.gnp_random_graph(n_nodes, p, directed=directed_test) for k in range(n_samples)]
nx_ER_end = time.time()

print("Elapsed time: {} s".format(nx_ER_end - nx_ER_start))
print("Produced {} samples".format(len(nx_ER_list)))


print("Now using ergmpy gibbs sampler ergm.sample_binary, same parameters")

ergm_ER_start = time.time()
ergm_ER_model = ergm.ergm([np.sum], [np.log(p / (1-p))], directed=directed_test)
ergm_ER_samples = ergm_ER_model.sample_binary(n_nodes, n_samples, verbose=2, block_size=3)
ergm_ER_end = time.time()

print("Elapsed time: {} s".format(ergm_ER_end - ergm_ER_start))

ergm_ER_list = [nx.from_numpy_array(ergm_ER_samples[:,:,i], create_using=nx.DiGraph) for i in range(n_samples)]

print("Produced {} samples".format(ergm_ER_samples.shape[2]))

print("Comparing distribution of edge counts:")

m = int(binom(n_nodes, 2)) * (1 + directed_test)  # should be 30.
# nx_edge_distro = [0.] * (m + 1)  # There are between 0 and 15 (inclusive) edges in each graph
# ergm_edge_distro = [0.] * (m + 1)
nx_edge_distro = np.zeros(m+1)
ergm_edge_distro = np.zeros(m+1)

theory_edge_distro = [binom(m,k) * (p ** k) * ((1 - p) ** (m - k)) for k in range(m+1)]
for (G_nx, k) in zip(nx_ER_list, range(n_samples)):
    nx_edge_distro[nx.number_of_edges(G_nx)] += 1
    ergm_edge_distro[np.sum(ergm_ER_samples[:, :, k])] += 1
nx_edge_distro = nx_edge_distro / n_samples
ergm_edge_distro = ergm_edge_distro / n_samples

# ergm_actual_edge_distroy = [np.sum(ergm_ER_samples[:,:,i]) for i in range(n_samples)]
# h = np.histogram(ergm_actual_edge_distroy, bins=m+1)

# nx_edge_distro = [d / sum(nx_edge_distro) for d in nx_edge_distro]
# ergm_edge_distro = [d / sum(ergm_edge_distro) for d in ergm_edge_distro]

print("{:>3} {:20} {:20} {:20}".format("m", "nx prob.", "ergm prob.", "theory prob."))
for d in range(m + 1):
    print(f"{d:3d} {nx_edge_distro[d]:20.14f} {ergm_edge_distro[d]:20.14f} {theory_edge_distro[d]:20.14f}")

print(f"avg {np.dot(nx_edge_distro, np.arange(m+1)):20.14f} {np.dot(ergm_edge_distro, np.arange(m+1)):20.14f} {np.dot(theory_edge_distro, np.arange(m+1)):20.14f}")
print()

n_large = 100
p_small = 0.05
n_samples = 1000
print("Now attempting {} samples from n = {} nodes, p = {}".format(n_samples, n_large, p_small))

nx_ER_large_start = time.time()
nx_ER_large_list = [nx.gnp_random_graph(n_large, p_small, directed=directed_test) for k in range(n_samples)]
nx_ER_large_end = time.time()
nx_ER_large_time = nx_ER_large_end - nx_ER_large_start
print("nx.gnp_random_graph took {} s".format(nx_ER_large_time))

nx_fastER_start = time.time()
nx_fastER_list = [nx.fast_gnp_random_graph(n_large, p_small, directed=directed_test) for k in range(n_samples)]
nx_fastER_end = time.time()
nx_fastER_time = nx_fastER_end - nx_fastER_start
print("nx.fast_gnp_random_graph took {} s".format(nx_fastER_time))

ergm_ER_large_start = time.time()
ergm_ER_large_model = ergm.ergm([np.sum], [math.log(p_small / (1-p_small))], directed=directed_test)
# ergm_ER_large_samples = ergm_ER_large_model.sample_binary(n_large,n_samples, burn_in=5*n_large, n_steps=2*n_large)
ergm_ER_large_samples = ergm_ER_large_model.sample_binary(n_large, n_samples, verbose=2, block_size=1)
ergm_ER_large_end = time.time()
ergm_ER_large_time = ergm_ER_large_end - ergm_ER_large_start
# print("ergm.sample_binary took {} s with {} burnin steps and {} steps between samples".format(ergm_ER_large_time, 5*n_large, 2*n_large))
print("ergm.sample_binary took {} s".format(ergm_ER_large_time))

nx_ER_avg = sum([nx.number_of_edges(G) for G in nx_ER_large_list]) / n_samples
nx_ER_fast_avg = sum([nx.number_of_edges(G) for G in nx_fastER_list]) / n_samples
ergm_ER_large_avg = sum([np.sum(ergm_ER_large_samples[:,:,k]) for k in range(n_samples)]) / n_samples
# theory_avg = binom(n_large, 2) * p_small
theory_avg = n_large * (n_large - 1) * p_small / (1 + (not directed_test))

print("Avg # of edges")
# print(f"{'theory',:20}{'nx.gnp':20}{'nx.fast_gnp':20}{'ergm':20}")
print(f"{'theory':20} {'nx.gnp':20} {'nx.fast_gnp':20} {'ergm':20}")
print(f"{theory_avg:20.10f} {nx_ER_avg:20.10f} {nx_ER_fast_avg:20.10f} {ergm_ER_large_avg:20.10f}")