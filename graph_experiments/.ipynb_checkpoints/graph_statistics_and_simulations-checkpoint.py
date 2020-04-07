import networkx as nx
import numpy as np
import time
import matplotlib.pyplot as plt
import random
    



#-------------- Functions that are available for graph stats ------------------ #
#adding attributes to functions
class run_options:
    def __init__(self, directional = False,multiedge = False):
        self.directional = directional
        self.multiedge = multiedge

    def __call__(self, f):
        f.directional = self.directional
        f.multiedge = self.multiedge
        return f
    

from networkx.algorithms import approximation as app

@run_options(directional=False,multiedge=False)
def n_triangles(current_graph):
    triangle_dict = nx.triangles(current_graph)
    n_triangles = np.sum(list(triangle_dict.values()))/3
    return n_triangles

@run_options(directional=False,multiedge=False)
def n_edges_empirical(current_graph):
    return len(current_graph.edges())


@run_options(directional=True,multiedge=False)
def transitivity(current_graph,**kwargs):
    """
    transitivity: Fraction of all possible traingles present in G
    Triad = 2 edges with a shared vertex

    Transitivity = 3* # of triangles/ # of traids

    """
    return nx.transitivity(current_graph)

@run_options(directional=True,multiedge=True)
def node_connectivity(current_graph,**kwargs):
    return app.node_connectivity(current_graph)

@run_options(directional=False,multiedge=True)
def size_maximum_clique(current_graph,**kwargs):
    """
    clique is just subset of vertices group where every
    vertex in group is connected (subgraph induced is complete)

    Maximum clique = clique of the largest size in a graph
    clique number = number of vertices in a maxium clique

    """
    return nx.graph_clique_number(current_graph)

@run_options(directional=False,multiedge=True)
def n_maximal_cliques(current_graph,**kwargs):
    """
    clique is just subset of vertices group where every
    vertex in group is connected (subgraph induced is complete)

    Maximal clique = clique that cannot be extended by including one or more adjacent vertex 
    (aka not subset of larger clique)
    Maximum clique = clique of the largest size in a graph
    clique number = number of vertices in a maxium clique

    """
    return nx.graph_number_of_cliques(current_graph)

@run_options(directional=True,multiedge=True)
def average_degree_connectivity(current_graph,**kwargs):
    """ Returns dictionary that maps nodes with a certain degree to the average degree of the nearest neightbors"""
    return nx.average_degree_connectivity(current_graph)


@run_options(directional=True,multiedge=False)
def average_clustering(current_graph,**kwargs):
    """ 
    local clustering: theoretically the fraction of traingles that actually exist / 
                                                    all possible traingles in its neighborhood
    How it is computed: 
    1) choose random node
    2) choose 2 neighbors at random
    3) check if traingle (if yes increment traingle counter)
    4) Repeat and compute number with triangle_counter/ trials

    """
    return nx.average_clustering(current_graph)

@run_options(directional=True,multiedge=True)
def min_weighted_vertex_cover_len(current_graph,**kwargs):
    """ 
    Returns length of Minimum number of vertices so that all edges are coincident on at least one vertice

    """
    return len(app.min_weighted_vertex_cover(current_graph))


@run_options(directional=False,multiedge=False)
def tree_number(current_graph,**kwargs):
    """ 
    Returns an approximation of the tree width of the graph (aka how tree-like it is):
    The lower the value the more tree-like the graph is

    """
    return app.treewidth_min_degree(current_graph)[0]

# ------ newly added statistics 3/19 ------------ #

@run_options(directional=True,multiedge=True)
def degree_distribution_mean(current_graph,**kwargs):
    sequences = [k for v,k in dict(current_graph.degree).items()]
    return np.mean(sequences)


@run_options(directional=True,multiedge=True)
def degree_distribution_mean(current_graph,**kwargs):
    sequences = [k for v,k in dict(current_graph.degree).items()]
    return np.mean(sequences)


@run_options(directional=False,multiedge=False)
def number_connected_components(current_graph,**kwargs):
    return nx.number_connected_components(current_graph)

@run_options(directional=False,multiedge=False)
def largest_connected_component_size(current_graph,**kwargs):
    Gcc = sorted(nx.connected_components(current_graph), key=len, reverse=True)
    return len(Gcc[0])


@run_options(directional=False,multiedge=False)
def largest_connected_component_size(current_graph,**kwargs):
    Gcc = sorted(nx.connected_components(current_graph), key=len, reverse=True)
    return len(Gcc[0])

# ------------- New Functions that were added 3/25 ------------------------ #
@run_options(directional=False,multiedge=False)
def inverse_average_shortest_path(G):
    Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
    sp = nx.average_shortest_path_length(nx.subgraph(G,Gcc[0]))
    if sp > 0:
        return 1/sp
    else:
        return None

    
    


# - For the percolation - #
def _get_vertex_order(G,selection_type="random"):
    if selection_type == "random":
        return np.random.permutation(list(G.nodes))
    elif selection_type == "degree":
        """ Will organize from highest to lowest degree"""
        degree_dict = dict()
        for k,v in G.degree():
            if v not in degree_dict.keys():
                degree_dict[v] = [k]
            else:
                degree_dict[v].append(k)
        degree_dict

        #get the order of degree
        order_degrees = np.sort(list(degree_dict.keys()))

        node_order = []
        for k in order_degrees:
            node_order += list(np.random.permutation(degree_dict[k]))

        return node_order
    else:
        raise Exception("Invalid Selection Type")

from tqdm import tqdm

def run_site_percolation(G,vertex_order_type="random",n_iterations=1000):
    total_runs = []

    for y in tqdm(range(0,n_iterations)):
        current_run_results = [0,1]
        """
        1) Start with empty network. Number of clusters, c = 0, currently in network
        Choose at random the order in which vertices will be added to the network
        """

        clusters=dict() #starting out the clusters list as empyt
        vertex_order = _get_vertex_order(G,vertex_order_type)


        """
        2) Add the next vertex in list to the network initially with no edges
        """
        vertex_labels = dict()
        for i,v in enumerate(vertex_order):
            #print(f"Working on vertex {v}")

            """ 2b)
            - increase the cluster count by 1 (because the new vertex is initially a cluster of its own)
            - Make the cluster size of one

            """

            try:
                max_index_plus_1 = np.max(list(clusters.keys())) + 1
                clusters[max_index_plus_1] = 1
                vertex_labels[v] = max_index_plus_1
            except:
                clusters[0] = 1
                vertex_labels[v] = 0
                continue

            """
            3) Go through the edges attached to newly added vertex and add the edges where the other 
            vertex already exists in the network

            4) For each edge added, check if the vertices have the same cluster group number:
            - if yes then do nothing
            - if no, relabel the smaller cluster the same cluster number as the bigger cluster number
            - update the sizes of the 2 clusters from which formed
            """
            already_added_v = set(vertex_order[:i]).intersection(set(G[v].keys()))
            for a_v in already_added_v:
                if vertex_labels[a_v] != vertex_labels[v]:
                    index_max = np.argmax([clusters[vertex_labels[a_v]],clusters[vertex_labels[v]]])
                    if index_max == 0: #need to change all the labels with v
                        replaced_cluster = vertex_labels[v]
                        indexes_to_change = [jj for jj in vertex_labels.keys() if vertex_labels[jj] == vertex_labels[v]]
                        final_cluster = vertex_labels[a_v]
                    else:
                        replaced_cluster = vertex_labels[a_v]
                        indexes_to_change = [jj for jj in vertex_labels.keys() if vertex_labels[jj] == vertex_labels[a_v]]
                        final_cluster = vertex_labels[v]

                    #change the labels
                    for vv in indexes_to_change:
                        vertex_labels[vv] = final_cluster

                    replaced_size = clusters.pop(replaced_cluster)
                    clusters[final_cluster] += replaced_size

            current_run_results.append(np.max([v for v in clusters.values()]))


            #Done adding that vertex and will continue on to next vertex
            #print(f"clusters = {clusters}")

            total_runs.append(current_run_results)
    total_runs = np.array(total_runs)
    
    from scipy.special import comb
    n = len(G.nodes)
    S_r = np.mean(total_runs,axis=0)
    #calculate s_phi : average largest cluster size as a functin of the occupancy probability
    phi = np.arange(0,1.05,0.05)
    r = np.arange(0,n+1,1)
    s_phi = [np.sum([comb(n, r_curr, exact=True)*(phi_curr**r_curr)*((1-phi_curr)**(n- r_curr))*S_r_curr
                        for r_curr,S_r_curr in zip(r,S_r)]) for phi_curr in phi]
    s_phi = np.array(s_phi)/n
    
    return s_phi,phi
    
import networkx as nx


@run_options(directional=False,multiedge=False)
def random_degree_site_percolation(G,n_iterations=200):
    random_degree_site_percolation.stat_names = ["area_above_identity_random_percol",
                                                "area_below_identity_random_percol",
                                                "area_above_identity_degree_percol",
                                                "area_below_identity_degree_percol"]
    s_phi_barabasi_rand,phi_barabasi_rand= run_site_percolation(G,"random",n_iterations)
    s_phi_barabasi_degree,phi_barabasi_degree= run_site_percolation(G,"degree",n_iterations)

    rand_diff = s_phi_barabasi_rand - phi_barabasi_rand
    degree_diff = s_phi_barabasi_degree - phi_barabasi_degree

    dx = phi_barabasi_rand[1]-phi_barabasi_rand[0]

    rand_diff_positive = np.where(rand_diff>0)[0]
    rand_diff_negative = np.where(rand_diff<= 0)[0]
    degree_diff_positive = np.where(degree_diff>0)[0]
    degree_diff_negative = np.where(degree_diff<=0)[0]

    return (np.trapz(rand_diff[rand_diff_positive],dx=dx),
     np.trapz(rand_diff[rand_diff_negative],dx=dx),
     np.trapz(degree_diff[degree_diff_positive],dx=dx),
     np.trapz(degree_diff[degree_diff_negative],dx=dx))

# - End of Percolation - #

# - Start of Beta Epidemic Stat -- #
    
import ndlib
import networkx as nx
import numpy as np

import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics as ep

def pandemic_beta_average(
                                graph,
                                average_iterations=5,
                                n_time_iterations = 200,
                                initial_infected_prop = 0.05,
                                gamma = 0.01,
                                beta_start = 0.00001,
                                current_jump=0.001,
                                pandemic_threshold = 0.7,
                                pandemic_dev = 0.01,
                                max_iterations = 50,
                                use_optimized_beta_finder=False
                              ):
    arg_dict = dict(
        n_time_iterations = n_time_iterations,
        initial_infected_prop = initial_infected_prop,
        gamma = gamma,
        beta_start = beta_start,
        current_jump=current_jump,
        pandemic_threshold = pandemic_threshold,
        pandemic_dev = pandemic_dev,
        max_iterations=max_iterations
            )
    
    pandemic_beta = []
    for i in range(0,average_iterations):
        print(f"\n    Working on Run {i}")
        percent_affected_history = []
        retry_counter = 0
        max_retry = 3
        while len(percent_affected_history)<= 1:
            percent_affected_history,beta_history = find_pandemic_beta(graph,**arg_dict)
            retry_counter += 1
            if retry_counter > max_retry:
                print(f"Could not find right Beta after {max_retry} tries: returning Last value hit: {beta_history[0]}")
                return beta_history[0]
                #raise Exception(f"Could not find right Beta after {max_retry} tries")
        optimal_beta = beta_history[-1]
        
        pandemic_beta.append(optimal_beta)
        """
        now adjust the beta_start and the current jump based on the history 
        before the next run
        
        Rule: Find the optimal beta
        starting_beta = optimal_beta - 3*last_jump
        
        
        """ 
        if use_optimized_beta_finder:
            if len(beta_history) >= 2:
                print(f"beta_history= {beta_history}")
                last_jump_size = np.abs(beta_history[-2] - beta_history[-1])
                arg_dict["beta_start"] = optimal_beta - 2*last_jump_size
                arg_dict["current_jump"] = 2*last_jump_size
            
        
    return np.mean(pandemic_beta)

def find_pandemic_beta(graph,
                       n_time_iterations = 200,
                        initial_infected_prop = 0.05,
                        gamma = 0.01,
                        beta_start = 0.00001,
                        current_jump=0.001,
                        pandemic_threshold = 0.7,
                        pandemic_dev = 0.01,
                       max_iterations=50
                       ):
    
    print_flag = False
    
    def _calculate_percent_affected(trends):
        n_recovered = trends[0]["trends"]["node_count"][2][-1]
        n_infected = trends[0]["trends"]["node_count"][1][-1]
        percent_affected = (n_recovered + n_infected)/len(graph.nodes)
        return percent_affected

    print(f"\n\n---- New Run: Finding Beta for [{pandemic_threshold - pandemic_dev}, {pandemic_threshold + pandemic_dev}]\n"
         f"    Starting with beta_start={beta_start},current_jump={current_jump}")
    percent_affected = 0
    counter = 0
    beta = beta_start
    
    beta_history=[]
    percent_affected_history = []
    
    
    while (percent_affected > pandemic_threshold + pandemic_dev
           or percent_affected < pandemic_threshold - pandemic_dev):
        if print_flag:
            print(f"Current loop {counter}")
        counter += 1
        if counter > max_iterations:
            print("Max iterations hit before convergence on Beta, going to try again")
            return [],[beta]

        model = ep.SIRModel(graph)
        #Setting the model configuration
        config = mc.Configuration()
        config.add_model_parameter('beta', beta)
        config.add_model_parameter('gamma', gamma)
        config.add_model_parameter("fraction_infected", initial_infected_prop) #not setting the initial nodes that are infected but just the initial fraction
        model.set_initial_status(config)

        # Simulation
        iterations = model.iteration_bunch(n_time_iterations) 
        trends = model.build_trends(iterations) # builds the  dict_keys(['node_count', 'status_delta']) time series
        percent_affected = _calculate_percent_affected(trends)

        beta_history.append(beta)
        percent_affected_history.append(percent_affected)
        
        if print_flag:
            print(f"With beta = {beta}, percent_affected = {percent_affected}, current_jump={current_jump}")
        #Adjust the Beta
        if percent_affected < pandemic_threshold - pandemic_dev:
            beta += np.min((1,current_jump))
        elif percent_affected > pandemic_threshold + pandemic_dev:
            #print("beta_history[-2] = " + str(beta_history[-2]))
            if percent_affected_history[-2] < pandemic_threshold - pandemic_dev: #if jumped over the answer
                if print_flag:
                    print("Jumped over answer")
                beta = np.max((beta - current_jump/2,0))
                current_jump = current_jump/2 - current_jump*0.01
                if current_jump < 0.000001:
                    current_jump = 0.000001
            else:
                beta = np.max((beta - current_jump,0))
            
        else:
            break
    return percent_affected_history,beta_history


@run_options(directional=False,multiedge=False)
def pandemic_beta(graph):
    print("\n\n-------Starting Pandemic Beta -------")
    print(f"n_edges = {len(graph.edges())}, n_nodes = {len(graph.nodes)}")
    start_time = time.time()
    final_beta_average = pandemic_beta_average(graph)
    print(f"final_beta_average = {final_beta_average}")
    print(f"Total time for optimized = {time.time() - start_time}\n")
    
    return final_beta_average



# - End of Beta Epidemic Stat -- #

import scipy

#eigenvalue measurements
@run_options(directional=False,multiedge=False)
def largest_adj_eigen_value(G1):
    Adj = nx.convert_matrix.to_numpy_matrix(G1)
    return np.real(np.max(np.linalg.eigvals(Adj)))

@run_options(directional=False,multiedge=False)
def smallest_adj_eigen_value(G1):
    Adj = nx.convert_matrix.to_numpy_matrix(G1)
    return np.real(np.min(np.linalg.eigvals(Adj)))

@run_options(directional=False,multiedge=False)
def largest_laplacian_eigen_value(G1):
    laplacian = scipy.sparse.csr_matrix.toarray(nx.laplacian_matrix(G1))
    return np.real(np.max(np.linalg.eigvals(laplacian)))

"""
SHOULDN'T REALLY HAVE TO USE THIS BECAUSE IT WILL ALWAYS BE 0
"""
@run_options(directional=False,multiedge=False)
def smallest_laplacian_eigen_value(G1):
    laplacian = scipy.sparse.csr_matrix.toarray(nx.laplacian_matrix(G1))
    return np.real(np.min(np.linalg.eigvals(laplacian)))

#*** THIS IS ALSO CALLED THE ALGEBRAIC CONNECTIVITY
@run_options(directional=False,multiedge=False)
def second_smallest_laplacian_eigen_value(G1):
    laplacian = scipy.sparse.csr_matrix.toarray(nx.laplacian_matrix(G1))
    sorted_eig_vals = np.sort(np.real(np.linalg.eigvals(laplacian)))
    return sorted_eig_vals[1]

from itertools import chain
@run_options(directional=False,multiedge=False)
def top_heavy_percentage(G1,top_percentage = 0.90):
    degree_sequence = np.array(G1.degree())[:,1]
    ordered_nodes = np.argsort(degree_sequence)


    index_to_start = np.ceil(len(degree_sequence)*top_percentage).astype("int")
    #print(f"index_to_start = {index_to_start}")
    top_nodes_to_keep = ordered_nodes[index_to_start:]
    #print(f"top_nodes_to_keep = {top_nodes_to_keep}")

    nodes_nbrs = G1.adj.items()
    top_neighbors = [set(v_nbrs) for v,v_nbrs in nodes_nbrs if v in top_nodes_to_keep]
    top_neighbors.append(set(top_nodes_to_keep))

    unique_top_neighbors = set(chain.from_iterable(top_neighbors))
    return len(unique_top_neighbors)/len(G1)

@run_options(directional=False,multiedge=False)
def critical_occupation_probability(G1):
    degree_sequence = np.array(G1.degree())[:,1]
    return np.mean(degree_sequence)/(np.mean(degree_sequence)**2 - np.mean(degree_sequence))


@run_options(directional=False,multiedge=False)
def rich_club_transitivity(G):
    """
    Computes the triad closure percentage between only those nodes with same or higher degree
    """
    nodes_nbrs = G.adj.items()

    triads = 0
    triangles = 0
    degree_lookup = dict(G.degree())

    for v,v_nbrs in nodes_nbrs:
        v_nbrs_degree = [vnb for vnb in v_nbrs if degree_lookup[vnb] >= degree_lookup[v]]
        vs=set(v_nbrs_degree)-set([v]) #getting all the neighbors of the node (so when put in different combinations these could be triads)
        local_triangles=0
        local_triads = len(vs)*(len(vs) - 1)
        if local_triads<1:
            #print("No local triads so skipping")
            continue
        for w in vs:
            ws = set(G[w])-set([w]) #gets all the neighbors of a neighbor except itself
            local_triangles += len(vs.intersection(ws)) #finds out how many common neighbors has between itself and main node

        #print(f"For neuron {v}: Triads = {local_triads/2}, Triangles = {local_triangles/2}, transitivity = {local_triangles/local_triads}")
        triads += local_triads 
        triangles+= local_triangles
    
    #print(f"Total: Triads = {triads/2}, Triangles = {triangles/2}, transitivity = {triangles/triads}")
    if triads > 0:
        return triangles/triads
    else:
        return None


# --- Powerlaw stats --- #

import powerlaw

def get_degree_distribution(G):
    return np.array(G.degree())[:,1]

@run_options(directional=False,multiedge=False)
def power_law_alpha_sigma(G):
    #get the degree distribution
    power_law_alpha_sigma.stat_names = ["power_law_alpha",
                                        "power_law_sigma"]
    fit = powerlaw.Fit(get_degree_distribution(G))
    return fit.power_law.alpha, fit.power_law.sigma

@run_options(directional=False,multiedge=False)
def power_law_sigma(G):
    #get the degree distribution

    fit = powerlaw.Fit(get_degree_distribution(G))
    return fit.power_law.sigma

@run_options(directional=False,multiedge=False)
def power_law_alpha(G):
    #get the degree distribution

    fit = powerlaw.Fit(get_degree_distribution(G))
    return fit.power_law.alpha

@run_options(directional=False,multiedge=False)
def power_exp_fit_ratio(G):
    """
    Will return the loglikelihood ratio of the power and exponential graph
    R:
    Will be positive if power is more likely
            negative    exponential
    
    p: significance of fit
    """
    #get the degree distribution
    power_exp_fit_ratio.stat_names = ["power_exp_LL_ratio",
                                        "power_exp_LL_ratio_sign"]
    
    fit = powerlaw.Fit(get_degree_distribution(G))
    R,p = fit.distribution_compare("power_law",
                                                 "exponential",
                                                normalized_ratio=True)
    return R

@run_options(directional=False,multiedge=False)
def trunc_power_stretched_exp_fit_ratio(G):
    """
    Will return the loglikelihood ratio of the power and exponential graph
    R:
    Will be positive if power is more likely
            negative    exponential
    
    p: significance of fit
    """
    #get the degree distribution
    trunc_power_stretched_exp_fit_ratio.stat_names = ["trunc_power_stretched_exp_LL_ratio",
                                        "trunc_power_stretched_exp_LL_ratio_sign"]
    
    fit = powerlaw.Fit(get_degree_distribution(G))
    R,p = fit.distribution_compare("truncated_power_law",
                                                 "stretched_exponential",
                                                normalized_ratio=True)
    return R


    

#-------------- Functions that are for running simulations ------------------ #


def apply_functions_vp2(my_func,G,my_args=dict(),
        time_print_flag=False,output_print_flag=False):
    """
    Purpose: Function that will take in statistic and graph and return the output
    
    Example of how to call one of the functions:
    import networkx as nx
    G = nx.fast_gnp_random_graph(10,0.4)
    di_G = nx.DiGraph(G)
    multi_G = nx.MultiGraph(G)

    current_function = tree_number
    my_args = dict(not_funny="hello")
    apply_functions_vp2(current_function,multi_G,my_args,time_print_flag=False,output_print_flag=False)
    apply_functions_vp2(current_function,di_G,my_args,time_print_flag=False,output_print_flag=False)
    
    """
    
    if time_print_flag:
        print(f"----- Working on {my_func.__name__}--------------")
    start_time = time.time()
    small_g_output = my_func(G,**my_args)
    if time_print_flag:
        print(f"    {my_func.__name__}  for graph with {len(G)} nodes took {time.time() - start_time}")
    if output_print_flag:
        print(f"{my_func.__name__}  for graph with {len(G)} nodes = {small_g_output}")
    
    #get the names of this this stat
    #print(f"small_g_output = {small_g_output}")
    if np.isscalar(small_g_output) or small_g_output==None:
        return {my_func.__name__:small_g_output}
    
    if len(small_g_output)<= 1:
        return {my_func.__name__:small_g_output[0]}
    else:
        #get the names of the outputs
        if "stat_names" in dir(my_func):
            if len(my_func.stat_names) != len(small_g_output):
                raise Exception("Number of outputs not match number of output names")
            return dict([(k,v) for k,v in zip(my_func.stat_names,small_g_output)])
            
        else:
            return {my_func.__name__:small_g_output} 
    
    return small_g_output



from networkx.algorithms import approximation as app
def graph_statistics_vp2(current_graph,functions_list,
                         my_args = dict(),
                     undirected_flag = False,
                     time_print_flag=False,
                     output_print_flag = False,
                    ):
    """
    Function capable of running a specified subset of the available statistis
    On a graph and then returning the results in a dictionary
    
    Example: 
    functions_list = [n_triangles,transitivity,
     node_connectivity,
     size_maximum_clique,
     n_maximal_cliques,
     average_degree_connectivity,average_clustering,
     min_weighted_vertex_cover_len,tree_number]

    stats_dict = graph_statistics_vp2(G,functions_list,
                                    my_args = dict(),
                                     undirected_flag = True,
                                     time_print_flag=False,
                                     output_print_flag = False)
    
    """
    if undirected_flag == True:
        current_graph = current_graph.to_undirected()
        
        
    """
    check the functions in the stats list match the type of graph:
    
    """
    multiedge_flag = False
    directional_flag = False
    if type(current_graph) == type(nx.MultiGraph()) or  type(current_graph) == type(nx.MultiDiGraph()):
        multiedge_flag=True
    if type(current_graph) == type(nx.DiGraph()) or  type(current_graph) == type(nx.MultiDiGraph()):
        directional_flag=True
        
    # get the list of functions that will work for this graph
    exclusion = []
    if multiedge_flag == True and directional_flag == False:
        exclusion = [k for k in functions_list if k.multiedge == False ]
    if multiedge_flag == False and directional_flag == True:
        exclusion = [k for k in functions_list if k.directional == False ]
    if multiedge_flag == True and directional_flag == True:
        exclusion = [k for k in functions_list if (k.directional == False or k.multiedge == False)]
        
    if len(exclusion) > 1:
        print(f"Excluding the following functions because they did not meet the directional or mutliedge capabilities that the input graph required: {[k.__name__ for k in exclusion]}")
    
    available_functions = [p for p in functions_list if p not in exclusion]
    
    #iterate throught the available functions and store the outputs:
    stats_dict = dict()
    
    for current_function in available_functions:
        stats_raw_dict = apply_functions_vp2(current_function,current_graph,
                                                        my_args=my_args,
                                                        time_print_flag=time_print_flag,
                                                        output_print_flag=output_print_flag)
        stats_dict.update(stats_raw_dict)
    return stats_dict




import itertools
def run_graph_experiment(graph_function,
                        functions_list,
                        iterable_loops,
                         n_iterations,
                         func_args = dict(),
                         loop_print_flag = False,
                        iterations_print_flag = False,
                         attr=[]
                        ):
    
    """
    Purpose: Will run the experiments on the graph type specified
    and return the results in a data structure like the following

    Data structure: dictionary with following

    key: graph name: [type]_[#nodes]-[#edges]-[#attr]-[#attr]
        example graph to plot
        n_iterations
        n_nodes
        n_edges
        parameters:
            parameter_value_1
            parameter_value_2
            etc...
        attribute list of strings
        graph_type
        iteration:
            stats dictionary:
                n_edges 
                stat_1
                stat_2....
                could be a histogram stored (like local centrality distribution)
                
    
    Example: 
    

    #calculates the statistics
    functions_list = [n_triangles,transitivity,
         node_connectivity,
         size_maximum_clique,
         n_maximal_cliques,
         min_weighted_vertex_cover_len,tree_number]

    graph_function = erdos_renyi_random_location

    attr=[]

    loop_print_flag = True
    iterations_print_flag = True

    loop_1 = dict(n=[10,20,50,100,150])
    loop_2 = dict(p=[0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4])
    iterable_loops = [loop_1,loop_2]
    n_iterations = 2


    data_structure = run_graph_experiment(graph_function,
                            functions_list,
                            iterable_loops,
                             n_iterations,
                             func_args = dict(),
                             loop_print_flag = loop_print_flag,
                            iterations_print_flag = iterations_print_flag,
                             attr=attr
                            )
            
    
    """


    stats_list = [k.__name__ for k in functions_list]

    loop_print_flag = loop_print_flag
    iteration_print_flag = iterations_print_flag

    #*************** EXTRACTS THE PARAMETER COMBINATIONS AND THE NAMES OF THE PARAMETERS ******************* #
    # get all the names of parameters stored in lists:
    parameter_names = [list(k.keys())[0] for k in iterable_loops]

    # initializing list of list  
    all_list = [k[list(k.keys())[0]] for k in iterable_loops]


    parameter_combinations = list(itertools.product(*all_list)) 

    # printing result 
    # print ("All possible permutations are : " +  str(res)) 
    # print("Parameter names = " + str(parameter_names))
    #print ("The original lists are : " + str(all_list)) 


    # *************** Settinup up datastructure to hold results ************
    total_stats_list = dict([(k,[]) for k in stats_list])
    print(f"graph_function = {graph_function}")
    graph_type = graph_function.__name__
    data_structure = dict()

    for current_parameter_combination in parameter_combinations:
        #print("current_parameter_combination = " + str(current_parameter_combination))
        #create dictionary with arguments
        graph_args = dict([(k,v) for k,v in zip(parameter_names,current_parameter_combination)])
        loop_start = time.time()
        for it in range(0,n_iterations):
            iteration_start = time.time()
            start_time = time.time()
            if iteration_print_flag:
                print(f"               Working on {[str(k) + '=' + str(v) for k,v in graph_args.items()]}, iter = {it}")
            creation_time = time.time()

            #creating the graph with randomize locations

            max_graph_creation_tries = 10
            creation_flag = False
            for i in range(0,max_graph_creation_tries):
                try:
                    print("graph_args = " + str(graph_args))
                    G = graph_function(**graph_args)
                    creation_flag = True
                    break
                except:
                    print(f"Graph creationg Failed, starting try #{i + 1}")

            if not creation_flag:
                raise Exception(f"Couldn't build graph after {max_graph_creation_tries} tries")
            
            

            if iteration_print_flag:
                print(f"               Graph Creation time = {time.time() - creation_time}")

            if it == 0:
                graph_name = str(graph_type) + ":"
                for k,v in graph_args.items():
                    if type(v) == float:
                        graph_name += f"_{k}={v:.2f}"
                    else:
                        graph_name += f"_{k}={v}"
                for at in attr:
                    graph_name += "_" + str(at)

                data_structure[graph_name] = dict()
                data_structure[graph_name]["n_edges"] = len(G.edges)
                data_structure[graph_name]["n_nodes"] = len(G.nodes)
                data_structure[graph_name]["iterations"] = dict()
                data_structure[graph_name]["n_iterations"] = n_iterations
                data_structure[graph_name]["attribute_list"] = attr
                data_structure[graph_name]["graph_type"] = graph_type
                data_structure[graph_name]["example_graph"] = G
                data_structure[graph_name]["parameters"] = dict()
                for k,v in graph_args.items():
                    data_structure[graph_name]["parameters"][k] = v

            stats_dict = dict()
            stats_dict["n_nodes"] = len(G.nodes)
            stats_dict["n_edges"] = len(G.edges)
            
            stats_dict_update = graph_statistics_vp2(G,functions_list,
                                            my_args = func_args,
                                             undirected_flag = False,
                                             time_print_flag=False,
                                             output_print_flag = False)

            stats_dict.update(stats_dict_update)
            data_structure[graph_name]["iterations"][it] = stats_dict

            if iteration_print_flag:
                print(f"               Total time for iteration = {time.time() - iteration_start}")
        if loop_print_flag:
                print(f"Total time for parameter loop = {time.time() - loop_start}")
    return data_structure
        


#----------------------- HOW TO SAVE OFF AND LOAD IN THE DATA STRUCTURE  ---------------------- #
import pandas as pd
def save_data_structure(path,file_name,data_structure):
    """
    Example: 
    save_data_structure(".","my_file",data_structure)
    """
    file_location_name = str(path) + "/" + str(file_name) + ".npz"
    np.savez(file_location_name,data_structure=data_structure,
            columns=data_structure.columns)
    

# Example of how to load saved data:
def load_data_structure(file_location_name):
    """ Example:
    data_structure = load_data_structure("my_file.npz")
    """
    er_total_stats_loaded = np.load(file_location_name,allow_pickle=True)
    er_data_structure = er_total_stats_loaded["data_structure"][()]
    new_df = pd.DataFrame(er_data_structure)
    new_df.columns = list(er_total_stats_loaded["columns"])
    return new_df

        
#  ------------------------------- HOW TO GENERATE TABLE WITH YOUR STATISTICS ------------------------ #
import pandas as pd
def table_comparison_statistics(data_structure,with_parameters=True,
                               statistics_list=[],
                               graph_lists=[]):

    """
    Purpose: return a pandas datatable with the statistics and graphs that you want
    
    -------Parameters (all of these parameters can be lists so can support multiple graph): ----------
    data_structure (dict) : dictionary holding all of the data
    statistics_list (list of str) : Speicifies the statisitcs we want to show up in the table
    graph_lists (list of str): Speicifies the graphs in the data_structure that we want to run
               
    ------  Returns: None ---------
    
    Pseudocode: 
    1) create new ditionary with all of the graph_lists we want
    2) For each statistic specified in the list go and extract it from our datacolumn
    3) Put into dictionary and then create pandas dataframe
    
    Example: 
    
    current_graph_list = [k for k in er_data_structure.keys() if "_20_" in k]
    current_stats_list = ['n_triangles', 'transitivity', 'node_connectivity', 'size_maximum_clique', 
                          'n_maximal_cliques', 'average_clustering', 'min_weighted_vertex_cover',]
                          #'tree_number', 'n_edges']

    current_return = table_comparison_statistics(er_data_structure,with_parameters=True,
                                                statistics_list = current_stats_list
                                                #graph_lists=current_graph_list
                                                )
    current_return
          
    """
    
    if len(graph_lists) == 0:
        graph_lists = list(data_structure.keys())
        
    iterations_flag = False
    if "iterations" in data_structure[graph_lists[0]].keys():
        iteration_flag = True
        
    if len(statistics_list) == 0:
        if iteration_flag == True:
            statistics_list = list(data_structure[graph_lists[0]]["iterations"][0].keys())
        else:
            statistics_list = list(data_structure[graph_lists[0]]["statistics"].keys())
    
    if with_parameters:
        #getting all the possible parameters in datastructure
        all_possible_parameters = []
        for g_type in data_structure.keys():
            all_possible_parameters += list(data_structure[g_type]["parameters"].keys())
        all_possible_parameters = list(set(all_possible_parameters))
        #print("all_possible_parameters = " + str(all_possible_parameters))
        statistics_list = all_possible_parameters + statistics_list
        
    #print(statistics_list)
    pd_data = dict(graphs=graph_lists)
    
    for stat in statistics_list:
        pd_data[stat] = []
        for g in graph_lists:
            #check if stat lives in the higher structure
            current_graph_stat = None
            if stat in data_structure[g].keys():
                current_graph_stat = data_structure[g][stat]
            elif stat in data_structure[g]["parameters"].keys():
                current_graph_stat = data_structure[g]["parameters"][stat]
                
            else:
                if iteration_flag == True:
                    if stat in data_structure[g]["iterations"][0].keys():
                    
                        current_graph_stat = np.nanmean(np.array([data_structure[g]["iterations"][i][stat] for i 
                                                           in data_structure[g]["iterations"].keys()]).astype("float"))
                else:
                    if stat in data_structure[g]["statistics"].keys():
                        current_graph_stat = data_structure[g]["statistics"][stat]
            pd_data[stat].append(current_graph_stat)
    pd_data["graph_type"] = [s[:s.index(":")] for s in graph_lists]
          
      
    
    #return pd.DataFrame.from_dict(pd_data).set_index("graphs",True)
    return pd.DataFrame.from_dict(pd_data)


# --------------------  The whole pandas to plots ---------------------- #
import pandasql
def restrict_pandas(df,index_restriction=[],column_restriction=[],value_restriction=""):
    """
    Pseudocode:
    How to specify:
    1) restrict rows by value (being equal,less than or greater) --> just parse the string, SO CAN INCLUDE AND
    2) Columns you want to keep
    3) Indexes you want to keep:
    
    Example: 
    
    index_restriction = ["n=50","p=0.25","erdos_renyi_random_location_n=100_p=0.10"]
    column_restriction = ["size_maximum_clique","n_triangles"]
    value_restriction = "transitivity > 0.3 AND size_maximum_clique < 9 "
    returned_df = restrict_pandas(df,index_restriction,column_restriction,value_restriction)
    returned_df
    
    Example of another restriction = "(transitivity > 0.1 AND n_maximal_cliques > 10) OR (min_weighted_vertex_cover_len = 18 AND size_maximum_clique = 2)"
    
    """
    new_df = df.copy()
    graph_name = "graph_name"
    if len(index_restriction) > 0:
        list_of_indexes = list(new_df[graph_name])
        restricted_rows = [k for k in list_of_indexes if len([j for j in index_restriction if j in k]) > 0]
        #print("restricted_rows = " + str(restricted_rows))
        new_df = new_df.loc[new_df[graph_name].isin(restricted_rows)]
        
    #do the sql string from function:
    if len(value_restriction)>0:
        s = ("SELECT * "
            "FROM new_df WHERE "
            + value_restriction + ";")
        
        #print("s = " + str(s))
        new_df = pandasql.sqldf(s, locals())
        
    #print(new_df)
    
    #restrict by the columns:
    if len(column_restriction) > 0:
        #column_restriction.insert(0,graph_name)
        new_df = new_df[column_restriction]
    
    return new_df




def plot_scalar_values(x,y=[],z=[],plot_type="scatter",flag_3D=False,
                       fig_ax=dict(),
                       return_fig_ax=False,
                       color_list=[],
                      label_list = []):
    """
    Purpose: To plot data in 3D or 2D with specified colors and labels:
    
    Example of How to Use:
    
    x =[1,2,3,4,5,6,7,8,9,10]
    x2 = [xl + 3 for xl in x]
    y =[5,6,2,3,13,4,1,2,4,8]
    y2 = [xl + 3 for xl in y]
    z =[2,3,3,3,5,7,9,11,9,10]
    z2 = [xl + 3 for xl in x]

    fig,ax = plot_scalar_values([x,x2],[y,y2],[z,z2],plot_type="scatter",flag_3D=True,return_fig=True,
                           color_list=["red","green"],
                          label_list = ["The only data","other_data"])
    ax.set_title("hello")
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.legend()
    
    """
    
    def change_to_list_of_lists(y):
        if any(isinstance(el, list) for el in y):
            return y
        else:
            return [y]
    
    from itertools import cycle
    cycol = cycle('bgrcmk')
    
#     print("before function")
#     print("x = " + str(x))
#     print("y = " + str(y))
#     print("z = " + str(z))
    x = change_to_list_of_lists(x)
    y = change_to_list_of_lists(y)
    z = change_to_list_of_lists(z)
#     print("AFTER function")
#     print("x = " + str(x))
#     print("y = " + str(y))
#     print("z = " + str(z))
    
#     print("x = " + str(len(x)))
#     print(len(y))
#     print(label_list)
    
    for x1,y1 in zip(x,y):
        if len(x1) != len(y1):
            raise Exception("x value dimensions do not match y dimensions")
    
    if len(fig_ax.keys()) <= 0:
        fig = plt.figure()
    else:
        fig = fig_ax["fig"]
        ax = fig_ax["ax"]
    if flag_3D:
        from mpl_toolkits.mplot3d import Axes3D
        if len(z)<=0:
            raise Exception("3D plotting specified but no z values recieved")
        for z1,x1 in zip(z,x):
            if len(z1) != len(x1):
                raise Exception("z value dimensions do not match x and y dimensions")
        if len(fig_ax.keys()) <= 0:
            ax = fig.add_subplot(111, projection='3d')
    else:
        if len(fig_ax.keys()) <= 0:
            ax = fig.add_subplot(111)
    
    #get the colors
    color_flag = False
    if len(color_list) > 0:
        if len(color_list) != len(x):
            raise Exception("Color list does not match dimensions of data")
        color_flag = True
    
    
    # get the labels and colors
    if len(label_list) <= 0:
        label_list = ["" for j in x]
    
    color_list_dict = []
    print("Color_lis = " + str(color_list))
    for i in range(0,len(label_list)):
        if color_flag == True:
            color_list_dict.append(dict(c=color_list[i]))
        else:
            color_list_dict.append(dict())
            
    print("color_list_dict = " + str(color_list_dict))
    
    #plot the values
    if flag_3D:
        for i,(x1,y1,z1) in enumerate(zip(x,y,z)):
            current_color = color_list_dict[i]
            #print(f"current_color = {current_color}")
            if plot_type == "line":
                ax.plot3D(x1, y1, z1, label=label_list[i],**color_list_dict[i])
            elif plot_type == "scatter":
                ax.scatter(x1, y1, z1, marker='o',label=label_list[i],**color_list_dict[i])
            else:
                raise Exception("Plot type is not valid")
    else:
        for i,(x1,y1) in enumerate(zip(x,y)):
            current_color = color_list_dict[i]
            #print(f"current_color = {current_color}")
            if plot_type == "line":
                ax.plot(x1, y1,label=label_list[i],**color_list_dict[i])
            elif plot_type == "scatter":
                ax.scatter(x1, y1, marker='o',label=label_list[i],**color_list_dict[i])
            else:
                raise Exception("Plot type is not valid")
                
                
    if return_fig_ax:
        print("returning figure and ax")
        return fig,ax
    else:
        fig.show()
        return


import itertools
def plot_df_values(df,
                   x_column,
                   y_column="",
                   index_restriction=[],
                   z_column="",
                   grouping=dict(),
                   color_list = [],
                   plot_type="scatter",
                   title="",
                  value_restriction="",
                  return_plotted_df_table=True,
                  return_fig_ax=False):
    """
    Will plot values from the data table as specified
    
    
    
    Pseudocode:
    Done in restrict pandas function
    1) Allow the rows to be restrictred (by name or by some value type)
    1b) Allow the y columns to be restricted (could do multiple)



    2) Allowed the type of plot to be specified (scatter, line) or specified as 3D
    3) Just the x,y,(z) values to be returned (so can do whatever want with plot), in dictionary
    4) Whether to return the figure
    5) Allow certain values to be grouped 
    
    
    How to specify grouping:
    dict( column: dict([exact,bin_equal_width,bin_equal_depth,string_of_divisors]) # but the exact is the only one that is implemented
    
    
    What we could want:
    1) Get what column you want to group by:
        a) If by class then just do type
        #b) if by variable then must give indexes or sql expressions (so check for strings or numbers)
        #c) LABELS DICTATED BY GROUPING?
    2) Specify the (x,y,) value or (x,y,z) values
    3) Specify the graphing type, title, colors,
    
    
    Example: 
    value_restriction = "transitivity > 0.3 AND size_maximum_clique < 9 "
    
    
    Example 1 on how to get the full visualization:
    returned_pandas = plot_df_values(df,index_restriction=[],
               x_column="n_triangles",
               y_column="transitivity",
               z_column = "size_maximum_clique",
                    grouping=dict(n=dict(exact=[]),),
                   plot_type="scatter",
                   title=" Looking at triangles",
                  value_restriction="tree_number > 4 AND n_triangles > 4",
                                )

    Example 2: 
    returned_pandas = plot_df_values(df,index_restriction=[],
               x_column="n_triangles",
               y_column="transitivity",
    #                    grouping=dict(n=dict(exact=[]),
    #                                 p=dict(exact=[])),
                        grouping=dict(n=dict(exact=[]),),
                       plot_type="line",
                       title=" Looking at triangles",
                      value_restriction="tree_number > 4 AND n_triangles > 4",
                                    )

    """
    
    #restrict the data table to what we want:
    
    #make the column restrictions only 
    possible_columns = list(df.columns)
    x_values = []
    if y_column != "":
        y_values = []
    if z_column != "":
        z_values = []
        
    
#     if sum([1 for k in column_restriction if k in possible_columns]) < len(column_restriction):
#         print("One or more of requested x,y,z values")
    
    returned_df = restrict_pandas(df,index_restriction,value_restriction=value_restriction)

    
    #now do the groupby to get the lists of our data
    """
    1) Assemble lists to iterate over in order to construct all of the mini tables
    2) iterat through list:
    a. Get the restricted table
    b. pull down the data requested
    c. add the data to a list so that can then send to the graphing function
    
    """
    list_of_lists = []
    parameter_names = list(grouping.keys())
    for k in parameter_names:
        if len(grouping[k]["exact"]) == 0:
            #get all of the possible unique values
            list_of_lists.append(list(set(returned_df[k].tolist())))
        else:
            list_of_lists.append(grouping[k]["exact"])
    
    #now have the keys and the lists to iterate over
    """
    Result will have lists that divide up your parameter space
    So can get every combination of that list
    Ex: 
    
    [['LPA_wheel',
  'LPA_random',
  'VD_mutation',
  'VD_basic',
  'VD_complement',
  'erdos_renyi',
  'power_law']]
    
    """
    
    parameter_combinations = list(itertools.product(*list_of_lists))
    
    

    #iterate through all the parameters and group by them
    labels_list = []
    fig_ax_dict = dict()
    
    frames = []
    
    for current_parameter_combination in parameter_combinations:
        group_args = dict([(k,v) for k,v in zip(parameter_names,current_parameter_combination)])
        #get the restricted datatable:
        if type(parameter_names[0]) == str:
            current_value_restriction = f"{parameter_names[0]} = '{current_parameter_combination[0]}'"
        else:
            current_value_restriction = f"{parameter_names[0]} = {current_parameter_combination[0]}"
           
        for i in range(1,len(parameter_names)):
            if type(parameter_names[0]) == str:
                current_value_restriction += f" AND {parameter_names[i]} = '{current_parameter_combination[i]}'"
            else:
                current_value_restriction += f" AND {parameter_names[i]} = {current_parameter_combination[i]}"
        
        
        #At this point have the certain grouping based on the restriction we want
        
        #print("current_value_restriction = " + str(current_value_restriction))
        new_table = restrict_pandas(returned_df,value_restriction=current_value_restriction)
        
        #check the length of the table
        if len(new_table) == 0:
            print(f"Parameter combination {group_args} had no dataframe values so continuing")
            continue
        
        
        labels_list.append(current_value_restriction)
        #get the values to pull down
        x_values.append(new_table[x_column].to_list())
        if y_column != "":
            y_values.append(new_table[y_column].to_list())
        if z_column != "":
            z_values.append(new_table[z_column].to_list())
        frames.append(new_table)
#     print("x_values = " + str(x_values))
#     print("y_values = " + str(y_values))
#     print("z_values = " + str(z_values))
    #print(labels_list)
    if y_column == "" and z_column == "":
        #plot in 1D
        pass
        
    
    if z_column != "":
        fig,ax = plot_scalar_values(x_values,y_values,z_values,plot_type=plot_type,flag_3D=True,fig_ax=fig_ax_dict,
                                    return_fig_ax=True,
                          label_list = labels_list,
                                   color_list=color_list)
    else:
        fig,ax = plot_scalar_values(x_values,y_values,plot_type=plot_type,flag_3D=False,fig_ax=fig_ax_dict,
                                    return_fig_ax=True,
                          label_list = labels_list,
                                   color_list=color_list)

    fig_ax_dict=dict(fig=fig,ax=ax) #don't need to pass this back anymore because only one loop
    
            

    
    ax.set_title(title)
    
    ax.set_xlabel(x_column)
    if y_column != "":
        ax.set_ylabel(y_column)
    if z_column != "":
        ax.set_zlabel(z_column)
        
        
    ax.legend()
    ax.get_legend().remove()
    leg = plt.legend( loc = 'upper right')

    # Get the bounding box of the original legend
    bb = leg.get_bbox_to_anchor().inverse_transformed(ax.transAxes)

    # Change to location of the legend. 
    xOffset = 1.2
    bb.x0 += xOffset
    bb.x1 += xOffset
    leg.set_bbox_to_anchor(bb, transform = ax.transAxes)    
        

    if return_fig_ax and return_plotted_df_table==False:
        return fig,ax
    elif return_fig_ax and return_plotted_df_table:
        return pd.concat(frames),(fig,ax)
    elif return_fig_ax == False and return_plotted_df_table:
        plt.show()
        return pd.concat(frames)
    else:
        plt.show()
    
#     if return_plotted_df_table:
#         #return returned_df
#         return pd.concat(frames)
