# functions for coloring networkx graphs
import numpy as np

def random_coloring(G,p0=None,p1=None,color_name='color'):
    """Applies a random coloring to graph `G`.

    `random_coloring(G, p0 = None, p1 = None, color_name = 'color')`

    `G` : a `networkx` graph(/digraph/multigraph/etc)

    `p0` : list of vertex color probabilities.
    If sum is less than one, last color has probability `1 - sum(p)`.
    If `p0 = None`, no color is applied.

    `p1` : Same, but for edge colors.

    `color_name` : What key to use for the color attribute. Default is `'color'`."""
    if p0 is not None:
        p0 = list(p0) + [1 - sum(p0)]
        k0 = np.random.choice(range(len(p0)), size=(len(G.nodes),), p=p0)
        for (v,k0v) in zip(G.nodes, k0):
            G.nodes[v][color_name] = k0v
    if p1 is not None:
        p1 = list(p1) + [1 - sum(p1)]
        k1 = np.random.choice(range(len(p1)),size=(len(G.edges),),p=p1)
        for (e,k1e) in zip(G.edges,k1):
            G[e[0]][e[1]][color_name] = k1e

def apply_coloring(G,k0=None,k1=None,color_name='color'):
    """Applies a coloring to vertices (specified by `k0`) and/or edges (specified by `k1`) of graph `G`.

    `G` : the graph

    `k0` : a function whose input is a vertex and output is a "color", meaning, for instance, a nonnegative integer.
    Strictly speaking, any output type that can be stored in a `dict` is fine.

    `k1` : a function whose input is two vertices (in order of directed edge, if edge is directed) and whose output is a color.

    `color_name` : the name for the color attribute. Default is `'color'`
    """
    if k0 is not None:
        for v in G.nodes:
            G.nodes[v][color_name] = k0(v)
    if k1 is not None:
        for e in G.edges:
            G[e[0]][e[1]][color_name] = k1(G.nodes[e[0]],G.nodes[e[1]])