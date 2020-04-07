import networkx as nx
import numpy as np
import time
import matplotlib.pyplot as plt
import random
    

def rename(newname):
    def decorator(f):
        f.__name__ = newname
        return f
    return decorator    

# ------------- BUILT IN GRAPH FUNCTIONS FROM NETWORKX ------------- #
@rename('erdos_renyi')
def erdos_renyi_random_location(n,p):
    """
    Erdos Renyi graph that has random locations generated
    """
    
    #erdos_renyi_random_location.graph_name = " erdos_renyi"
    
    network = nx.fast_gnp_random_graph(n,p)
    
    #setting random node locations
    node_locations =np.array([[random.uniform(0, 1),
                   random.uniform(0, 1),
                   random.uniform(0, 1)] for i in range(0,n)])

    nx.set_node_attributes(network, dict([(i,node_locations[i,:]) 
                              for i,node in enumerate(network.nodes)]), 'locations')
    
    return network

@rename('small_world')
def watts_strogatz_graph_smallworld_random_location(n,m,p):
    #watts_strogatz_graph_smallworld_random_location.__name__ = " small_world"
    
    network = nx.generators.random_graphs.watts_strogatz_graph(n=n,k=m,p=p)
    
    
    
    #setting random node locations
    node_locations =np.array([[random.uniform(0, 1),
                   random.uniform(0, 1),
                   random.uniform(0, 1)] for i in range(0,n)])

    nx.set_node_attributes(network, dict([(i,node_locations[i,:]) 
                              for i,node in enumerate(network.nodes)]), 'locations')
    
    return network

@rename('random_tree')
def random_tree_random_location(n):
    #random_tree_random_location.__name__ = "random_tree"
    network = nx.generators.trees.random_tree(n)
    
    #setting random node locations
    node_locations =np.array([[random.uniform(0, 1),
                   random.uniform(0, 1),
                   random.uniform(0, 1)] for i in range(0,n)])

    nx.set_node_attributes(network, dict([(i,node_locations[i,:]) 
                              for i,node in enumerate(network.nodes)]), 'locations')
    return network


@rename('random_uniform')
def random_uniform(n,m):
    #random_uniform.__name__ = "random_uniform"
    return nx.random_regular_graph(m,n)


# ------------- SPECIFYING A CERTAIN DEGREE SEQUENCE ---------------- #

def power_law_sequence(n,alpha,xmin,before_xmin_func=None,perc_before_xmin=0,before_xmin_func_args=dict()):
    if perc_before_xmin <= 0:
        r = np.random.uniform(0,1,n)
        x = xmin * (1-r) ** (-1/(alpha-1))
        return x
    else:
        n_before = np.floor(perc_before_xmin*n)
        n_after = n - n_before
        
        r = np.random.uniform(0,1,n)
        x_after = xmin * (1-r) ** (-1/(alpha-1))
        
        x_before = before_xmin_func(before_xmin_func_args.update(dict(n=n_before)))
        
        x_total = np.hstack([x_before,x_after])
        return x_total
        
from collections import Counter

def uniform_sequence(n,k_max):
    """
    Return sample sequence of length n from a uniform distribution.
    """
    #print(f"k_max = {k_max}")
    #uniform_seq = np.random.uniform(0,k_max,n)
    uniform_seq = np.random.randint(low = 1, high = k_max, size = n) 
    #print(f"Sequence about to be returned = {Counter(uniform_seq)}")
    return uniform_seq


def create_degree_sequence(n, sfunction=None, max_tries=200, **kwds):
    """ Attempt to create a valid degree sequence of length n using
    specified function sfunction(n,**kwds).

    Parameters
    ----------
    n : int
        Length of degree sequence = number of nodes
    sfunction: function
        Function which returns a list of n real or integer values.
        Called as "sfunction(n,**kwds)".
    max_tries: int
        Max number of attempts at creating valid degree sequence.

    Notes
    -----
    Repeatedly create a degree sequence by calling sfunction(n,**kwds)
    until achieving a valid degree sequence. If unsuccessful after
    max_tries attempts, raise an exception.
    
    For examples of sfunctions that return sequences of random numbers,
    see networkx.Utils.

    Examples
    --------
    >>> from networkx.utils import uniform_sequence, create_degree_sequence
    >>> seq=create_degree_sequence(10,uniform_sequence)
    """
    tries=0
    max_deg=n
    while tries < max_tries:
        if tries % 10:
            print(f"     Still working: On iteration number {tries}")
        trialseq=sfunction(n=n,**kwds)
        # round to integer values in the range [0,max_deg]
        seq=[min(max_deg, max( int(round(s)),0 )) for s in trialseq]
        #if graphical return, else throw away and try again
        #if sum(seq) % 2 == 0:
        if nx.is_graphical(seq):
            print("returning a sequence that can be exactly build")
            return seq,True
        tries+=1
    tries = 0
    while tries < max_tries:
        trialseq=sfunction(n=n,**kwds)
        # round to integer values in the range [0,max_deg]
        seq=[min(max_deg, max( int(round(s)),0 )) for s in trialseq]
        #if graphical return, else throw away and try again
        if sum(seq) % 2 == 0:
        #if nx.is_graphical(powerlaw_degree_seq):
            print("Couldn't find exact sequence, return even sequence")
            return seq,False
        tries+=1
    raise nx.NetworkXError(\
          "Exceeded max (%d) attempts at a valid sequence."%max_tries)

def random_sequence_law(n,**kwargs):
    """
    Will create a random graph based on the sequence function you 
    pass it and other parameters
    
    Example use: 
    random_sequence_law(n=10,sfunction=power_law_sequence, max_tries=50, xmin = 1,alpha=2)
    
    """
    #print(f"inside random_sequence_law for function = {kwargs['sfunction']}")
    degree_seq,exact_sequence_flag = create_degree_sequence(n = n, **kwargs)
    #print(f"degree_seq before configuration = {Counter(degree_seq)}")
    
    #print("USING THE CONFIGURATION MODEL")
    if not exact_sequence_flag:
        
        G=nx.configuration_model(degree_seq)
    else:
        #G = nx.generators.degree_seq.random_degree_sequence_graph(degree_seq)
        #G = nx.generators.degree_seq.expected_degree_graph(degree_seq)
        G=nx.configuration_model(degree_seq)
        
    #print(f"Before fixing the self-loops,multi-edges, degree sequence after configuration is = {Counter([v for k,v in dict(G.degree()).items()])}")
    G=nx.Graph(G) # to remove parallel edges
    try:
        G.remove_edges_from(G.selfloop_edges()) # to remove self loops
        return G
    except:
        return G

    
@rename('power_law')
def random_power_law(n,alpha,xmin=3):
    #random_power_law.__name__ = "power_law"
    G = random_sequence_law(n = n,
                              sfunction=power_law_sequence,
                                 alpha=alpha,
                                 xmin=xmin)
    return G
    
    
# ----------------Those functions that rely on growing mechanism -------- #

# ------- preferential Attachment ----- #
@rename('LPA_random')
def linear_preferencial_attachment_random(n,m,p=0.5,G = None,seed=None):
    #linear_preferencial_attachment_random.__name__ = "LPA_random"
    """
    Will generate a rich get richer graph
    
    preferential attachment but starts from a random graph
    
    Example: 
    G_LPA = linear_preferencial_attachment_random(20,5,0.3)
    nx.draw(G_LPA)
    
    m = number of edges you want per node
    p = the probability of connection at the start
    
    """
    if not G:
        #print("using erdos renyi model")
        G = nx.erdos_renyi_graph(m,p)
    #print(len(G.edges()))
    targets = np.array(G.edges()).ravel()
    i = m
    #loop that will add the node
    while i < n:
        #print(f"targets = {targets}, m = {m}")
        new_targets = np.random.choice(targets,m)
        #print(f"new_targets = {new_targets}")
        G.add_edges_from(zip([i]*m,new_targets))
        #print(f"G.edges = {G.edges}")
        targets = np.array(G.edges()).ravel()
        #targets = np.hstack([targets,new_targets])

        i += 1

    return G



def _random_subset(seq,m):
    """ Return m unique elements from seq.

    This differs from random.sample which can return repeated
    elements if seq holds repeated elements.
    """
    targets=set()
    while len(targets)<m:
        x=random.choice(seq)
        targets.add(x)
    return targets

from networkx.generators.classic import empty_graph, path_graph, complete_graph

@rename('LPA_wheel')
def linear_preferncial_attachment_wheel(n, m, seed=None):
    """Returns a random graph according to the Barabási–Albert preferential
    attachment model. Preferential Attachment but starts from spoke and wheel graph

    A graph of ``n`` nodes is grown by attaching new nodes each with ``m``
    edges that are preferentially attached to existing nodes with high degree.

    Parameters
    ----------
    n : int
        Number of nodes
    m : int
        Number of edges to attach from a new node to existing nodes
    seed : int, optional
        Seed for random number generator (default=None).

    Returns
    -------
    G : Graph

    Raises
    ------
    NetworkXError
        If ``m`` does not satisfy ``1 <= m < n``.

    References
    ----------
    .. [1] A. L. Barabási and R. Albert "Emergence of scaling in
       random networks", Science 286, pp 509-512, 1999.
    """
    
    
    
    #linear_preferncial_attachment_wheel.__name__ = "LPA_wheel"

    if m < 1 or  m >=n:
        raise nx.NetworkXError("Barabási–Albert network must have m >= 1"
                               " and m < n, m = %d, n = %d" % (m, n))
    if seed is not None:
        random.seed(seed)

    # Add m initial nodes (m0 in barabasi-speak)
    G=empty_graph(m)
    G.name="barabasi_albert_graph(%s,%s)"%(n,m)
    # Target nodes for new edges
    targets=list(range(m))
    # List of existing nodes, with nodes repeated once for each adjacent edge
    repeated_nodes=[]
    # Start adding the other n-m nodes. The first node is m.
    source=m
    while source<n: #because want to get all the way to n nodes and starting with m
        # Add edges to m nodes from the source to all of the targets
        G.add_edges_from(zip([source]*m,targets))
        # Add one node to the list for each new edge just created.
        repeated_nodes.extend(targets)
        # And the new node "source" has m edges to add to the list.
        repeated_nodes.extend([source]*m)
        # Now choose m unique nodes from the existing nodes
        # Pick uniformly from repeated_nodes (preferential attachement)
        targets = _random_subset(repeated_nodes,m)
        #print(f"targets = {targets}")
        source += 1
    return G





# -------------------- Vertex Duplication ----------------------------------- #

# BASIC vertex duplication:

import random

@rename('VD_basic')
def vertex_duplication(n, p, seed=None):
    """Returns an undirected graph using the duplication-divergence model.

    A graph of ``n`` nodes is created by duplicating the initial nodes
    and retaining edges incident to the original nodes with a retention
    probability ``p``.

    Parameters
    ----------
    n : int
        The desired number of nodes in the graph.
    p : float
        The probability for retaining the edge of the replicated node.
    seed : int, optional
        A seed for the random number generator of ``random`` (default=None).

    Returns
    -------
    G : Graph

    Raises
    ------
    NetworkXError
        If `p` is not a valid probability.
        If `n` is less than 2.
        
        
    Example: nx.draw(vertex_duplication(20,0.5))

    """
    #vertex_duplication.__name__ = "VD_basic"
    
    if p > 1 or p < 0:
        msg = "NetworkXError p={0} is not in [0,1].".format(p)
        raise nx.NetworkXError(msg)
    if n < 2:
        msg = 'n must be greater than or equal to 2'
        raise nx.NetworkXError(msg)
    if seed is not None:
        random.seed(seed)

    G = nx.Graph()
    G.graph['name'] = "Duplication-Divergence Graph"

    # Initialize the graph with two connected nodes.
    G.add_edge(0,1)
    i = 2
    while i < n:
        # Choose a random node from current graph to duplicate.
        random_node = np.random.choice(G.nodes())
        # Make the replica.
        G.add_node(i)
        # flag indicates whether at least one edge is connected on the replica.
        flag=False
        for nbr in G.neighbors(random_node):
            if random.random() < p:
                # Link retention step.
                G.add_edge(i, nbr)
                flag = True

        
        # NOT GOING TO DELETE NODE FROM THE GROUP IF IT DOES NO EDGES
        if not flag:
            # Delete replica if no edges retained.
            G.remove_node(i)
        else:
            # Successful duplication.
            i += 1

    return G

# vertex duplication WITH RANDOM MUTATION:
"""
How it works: 
1) Start with 2 nodes connected
2) Adds on node and decides to replicate

"""

import random

@rename('VD_mutation')
def vertex_duplication_with_mutation(n, p, p2, seed=None):
    """Returns an undirected graph using the duplication-divergence model.
    with probability p copies the edges of a node
    with probability (1-p makes random edge)

    """
    #vertex_duplication_with_mutation.__name__ = "VD_mutation"
    
    p_mut = p2
    if p > 1 or p < 0:
        msg = "NetworkXError p={0} is not in [0,1].".format(p)
        raise nx.NetworkXError(msg)
    if n < 2:
        msg = 'n must be greater than or equal to 2'
        raise nx.NetworkXError(msg)
    if seed is not None:
        random.seed(seed)

    G = nx.Graph()
    G.graph['name'] = "Duplication-Divergence Graph (with Mutation)"

    # Initialize the graph with two connected nodes.
    G.add_edge(0,1)
    i = 2
    
    while i < n:
        # Choose a random node from current graph to duplicate.
        
        random_node = np.random.choice(G.nodes())
        # Make the replica.
        G.add_node(i)
        # flag indicates whether at least one edge is connected on the replica.
        flag= False
        random_node_neighbors = list(G.neighbors(random_node)).copy()
        for nbr in random_node_neighbors:
            if random.random() < p:
                # Link retention step.
                G.add_edge(i, nbr)
                flag = True
        """
        Alteration: 
        1) For all neighbors not connected to new node
        2) connect to them with probability p_mut/(i-1), so that it decays over time
        """
        
        #get all of the neighbors not connected to 
        current_neighbors = set(G.neighbors(i))
        remaining_neighbors = set(G.nodes()).difference(current_neighbors)
        
        for nbr in remaining_neighbors:
            if random.random() < p_mut/(i-1):
                # Link retention step.
                G.add_edge(i, nbr)
                flag = True
        
        

        # NOT GOING TO DELETE NODE FROM THE GROUP IF IT DOES NO EDGES
        if not flag:
            # Delete replica if no edges retained.
            G.remove_node(i)
        else:
            # Successful duplication.
            i += 1
            
        
    return G


import random

@rename('VD_complement')
def vertex_duplication_with_complement(n, p, p2, seed=None):
    """Returns an undirected graph using the duplication-divergence model.
    with probability p copies the edges of a node
    with probability (1-p makes random edge)

    """
    
    #vertex_duplication_with_complement.__name__ = "VD_complement"
    p_con = p2
    if p > 1 or p < 0:
        msg = "NetworkXError p={0} is not in [0,1].".format(p)
        raise nx.NetworkXError(msg)
    if n < 2:
        msg = 'n must be greater than or equal to 2'
        raise nx.NetworkXError(msg)
    if seed is not None:
        random.seed(seed)

    G = nx.Graph()
    G.graph['name'] = "Duplication-Divergence Graph (with Mutation)"

    # Initialize the graph with two connected nodes.
    G.add_edge(0,1)
    i = 2
    while i < n:
        # Choose a random node from current graph to duplicate.
        #print("G.nodes() = " + str(G.nodes()))
        random_node = np.random.choice(G.nodes())
        #print(f"Random Node = {random_node}")
        # Make the replica.
        G.add_node(i)
        # flag indicates whether at least one edge is connected on the replica.
        #print("G.nodes() = " + str(G.nodes()))
        flag= False
        random_node_neighbors = list(G.neighbors(random_node)).copy()
        #print(f"random_node_neighbors = {random_node_neighbors}")
        for nbr in random_node_neighbors:
            if random.random() < p:
                # Link retention step.
                #print(f"copying {random_node} - {nbr}")
                G.add_edge(i, nbr)
                flag = True
            else:
                #decide with 50/50 shot of whether to add edge to new vertices
                #and delete the copied edge or keep everything the same
                if np.random.choice([0,1]):
                    #print(f"complementation {random_node} - {nbr}")
                    flag = True
                    G.remove_edge(random_node,nbr)
                    G.add_edge(i,nbr)
                
                    
        #decide whether to connect the new vertex and the one we copied

        
        if np.random.choice([0,1],p=[(1-p_con),p_con]):
            G.add_edge(random_node,nbr)
            flag = True

        
#         # NOT GOING TO DELETE NODE FROM THE GROUP IF IT DOES NO EDGES
#         if not flag:
#             # Delete replica if no edges retained.
#             G.remove_node(i)
            
            
        #delete all nodes with 0
        for nd in [k for k,v in G.degree() if v == 0]:
            G.remove_node(nd)
            
        mapping = dict([(yi,i) for i,yi in enumerate(G.nodes()) ])
        G = nx.relabel_nodes(G, mapping)
        
        #print(G.degrees())
        i = len(G.nodes())
        
    return G
