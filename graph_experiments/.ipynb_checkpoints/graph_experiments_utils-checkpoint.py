import networkx as nx
import numpy as np
import time
import matplotlib.pyplot as plt
import random
    


#-------------- Functions available for creating graphs -----------------------#
def erdos_renyi_random_location(n,p):
    """
    Erdos Renyi graph that has random locations generated
    """
    
    network = nx.fast_gnp_random_graph(n,p)
    
    #setting random node locations
    node_locations =np.array([[random.uniform(0, 1),
                   random.uniform(0, 1),
                   random.uniform(0, 1)] for i in range(0,n)])

    nx.set_node_attributes(network, dict([(i,node_locations[i,:]) 
                              for i,node in enumerate(network.nodes)]), 'locations')
    
    return network

def watts_strogatz_graph_smallworld_random_location(n,p,k):
    network = nx.generators.random_graphs.watts_strogatz_graph(n,k,p)
    
    #setting random node locations
    node_locations =np.array([[random.uniform(0, 1),
                   random.uniform(0, 1),
                   random.uniform(0, 1)] for i in range(0,n)])

    nx.set_node_attributes(network, dict([(i,node_locations[i,:]) 
                              for i,node in enumerate(network.nodes)]), 'locations')
    
    return network
   
def random_tree_random_location(n):
    network = nx.generators.trees.random_tree(n)
    
    #setting random node locations
    node_locations =np.array([[random.uniform(0, 1),
                   random.uniform(0, 1),
                   random.uniform(0, 1)] for i in range(0,n)])

    nx.set_node_attributes(network, dict([(i,node_locations[i,:]) 
                              for i,node in enumerate(network.nodes)]), 'locations')
    return network

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
        stats_dict[current_function.__name__] = apply_functions_vp2(current_function,current_graph,
                                                        my_args=my_args,
                                                        time_print_flag=time_print_flag,
                                                        output_print_flag=output_print_flag)
        
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
            G = graph_function(**graph_args)

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
def save_data_structure(path,file_name,data_structure):
    """
    Example: 
    save_data_structure(".","my_file",data_structure)
    """
    file_location_name = str(path) + "/" + str(file_name) + ".npz"
    np.savez(file_location_name,data_structure=data_structure)
    

# Example of how to load saved data:
def load_data_structure(file_location_name):
    """ Example:
    data_structure = load_data_structure("my_file.npz")
    """
    er_total_stats_loaded = np.load(file_location_name,allow_pickle=True)
    er_data_structure = er_total_stats_loaded["data_structure"][()]
    return er_data_structure






        
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
                    
                        current_graph_stat = np.mean([data_structure[g]["iterations"][i][stat] for i 
                                                           in data_structure[g]["iterations"].keys()])
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
    if len(index_restriction) > 0:
        list_of_indexes = list(new_df["graphs"])
        restricted_rows = [k for k in list_of_indexes if len([j for j in index_restriction if j in k]) > 0]
        #print("restricted_rows = " + str(restricted_rows))
        new_df = new_df.loc[new_df["graphs"].isin(restricted_rows)]
        
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
        column_restriction.insert(0,"graphs")
        new_df = new_df[column_restriction]
    
    return new_df




def plot_scalar_values(x,y,z=[],plot_type="scatter",flag_3D=False,
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
    
    
    # get the labels
    if len(label_list) <= 0:
        label_list = ["" for j in x]
    
    color_dict=dict()
    if color_flag == True:
        color_dict["c"] = color_list[i]
    
    #plot the values
    if flag_3D:
        for i,(x1,y1,z1) in enumerate(zip(x,y,z)):
            if plot_type == "line":
                ax.plot3D(x1, y1, z1, label=label_list[i],**color_dict)
            elif plot_type == "scatter":
                ax.scatter(x1, y1, z1, marker='o',label=label_list[i],**color_dict)
            else:
                raise Exception("Plot type is not valid")
    else:
        for i,(x1,y1) in enumerate(zip(x,y)):
            if plot_type == "line":
                ax.plot(x1, y1,label=label_list[i],**color_dict)
            elif plot_type == "scatter":
                ax.scatter(x1, y1, marker='o',label=label_list[i],**color_dict)
            else:
                raise Exception("Plot type is not valid")
    if return_fig_ax:
        print("returning figure and ax")
        return fig,ax
    else:
        fig.show()
        return


import itertools
def plot_df_values(df,index_restriction,
                   x_column,y_column,z_column="",
                   grouping=dict(),
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
    if z_column == "":
        #column_restriction = [x_column,y_column]
        x_values = []
        y_values = []
    else:
        #column_restriction = [x_column,y_column,z_column]
        x_values = []
        y_values = []
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
            append(grouping[k]["exact"])
    
    #now have the keys and the lists to iterate over
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
        
        
        #print("current_value_restriction = " + str(current_value_restriction))
        new_table = restrict_pandas(returned_df,value_restriction=current_value_restriction)
        
        #check the length of the table
        if len(new_table) == 0:
            print(f"Parameter combination {group_args} had no dataframe values so continuing")
            continue
        
        
        labels_list.append(current_value_restriction)
        #get the values to pull down
        x_values.append(new_table[x_column].to_list())
        y_values.append(new_table[y_column].to_list())
        if z_column != "":
            z_values.append(new_table[z_column].to_list())
        frames.append(new_table)
#     print("x_values = " + str(x_values))
#     print("y_values = " + str(y_values))
#     print("z_values = " + str(z_values))
    #print(labels_list)
    if z_column != "":
        fig,ax = plot_scalar_values(x_values,y_values,z_values,plot_type=plot_type,flag_3D=True,fig_ax=fig_ax_dict,
                                    return_fig_ax=True,
                          label_list = labels_list)
    else:
        fig,ax = plot_scalar_values(x_values,y_values,plot_type=plot_type,flag_3D=False,fig_ax=fig_ax_dict,
                                    return_fig_ax=True,
                          label_list = labels_list)

    fig_ax_dict=dict(fig=fig,ax=ax) #don't need to pass this back anymore because only one loop
    
            

    
    ax.set_title(title)
    
    ax.set_xlabel(x_column)
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



# --------------------  END OF PLOTTING WITH PANDAS ---------------------- #
























































































#-----------------   THE GRAPHS ------------------------------ #

















#----------------------- How to visualize graphs saved in the datastructure ---------------------- 




import matplotlib

# how to draw in 3D using ipyvolume
import ipyvolume as ipv
import numpy as np

import networkx as nx
import matplotlib.pyplot as plt
import random
from grave import plot_network, style_merger


import matplotlib

def graph_network_matplotlib(G,plot_type="2D",layout="non_random",locations=None,
                 colors=None,
                 default_color="blue",
                  colors_edge = None,
                  default_color_edge = "green",
                 plot_title="",
                 return_fig = False,
                             node_size=64,
                            node_label_size=20):
    """
    Purpose: To graph a network in either 2D or 3D with 
    optional specifications for different colors and different layouts
    
    -------Parameters (all of these parameters can be lists so can support multiple graph): ----------
    G (nx.Graph) : The Graph that will be plotted
    plot_type (str) : Specifies whether you want a 2D or 3D graph
    layout (np.array of str): Either a word description of how the nodes should be layed out in 2D (random or projection)
                            Or cartesian/random in 3D
    
    *locations (dict) : Overrides the existing locations that may be store in the graph structure
                            as node attributes
                            
    *color_list (dict/str) : Specifies the colors of each node to be plotted 
                            (could be string and just set as "location" to have as default
                            or "color" name to specify colors for all)
                            
                            
    ------  Returns: None ---------
    
    Pseudocode: 
    1) Identify how many graphs have been passed and start loop that will go through all graphs
        - if multiply graphs then verify that plot type is all the same if it is a list
        
    2) check if color list specified: (identify as list or scalar --> turn to list)
        a. If yes --> make sure that it matches the number of nodes for all the graphs 
                    (if "locations" or "color" specified then create list for that)
        b. If no --> check for color attributes within the nodes of the graph
            i. if yes --> extract those as the color list
            ii. if no --> generate random color list for nodes
    3) Identify the locations vector specified: (identify as list or scalar --> turn to list)
        - if none in either override or graph then just save as empty array
    4) Identify the type of plot wanted (identify as list or scalar --> turn to list)
                ****** for 2 - 4, if a list was specified but not the same length as Graph 
                        (and not 1) then raise exception *******************
                        
    a. If 2D type: get the layout type:
        If random --> use grave with no color specifications (and specify colors)
        Others: 
        If projection --> use grave put specify the correct projection (and specify colors)
    b. If 3D type: 
        if random --> generate random locations and then plot using ipyvolume
        if blank or cartesian --> use the locations specified in graph or locations
        
    5) Continue for all graphs
    6) issue the plot command
        
        
        
    Example on how to use:
    
    # print("\n\n\n*************** SIMPLE GRAPH *********************")
    #Simple graph

    return_value = graph_network(current_graph_example,plot_type="3D",layout="random",
                                              colors="pink")



    #Specifying different colors and locations

    # How to specify new locations

    variable_locations =np.array([[random.uniform(0, 1),
                               random.uniform(0, 1),
                               random.uniform(0, 1)] for i in range(0,len(current_graph_example.nodes))])

    celii_locations = dict([(n,vl) for n,vl in zip(current_graph_example.nodes,variable_locations)])

    # How to specify new colors with dictionary
    print("\n\n\n*************** DICTATING THE COLORS AND DICTIONARIES  *********************")

    celii_colors= dict([(n,vl) for n,vl in zip(current_graph_example.nodes,variable_locations)])



    return_value = graph_network(current_graph_example,plot_type="3D",
                                                layout="collapse_z",
                                               #locations = celii_locations,
                                              colors="locations",
                                             #colors = celii_colors
                                                )


    print("\n\n\n*************** OVERLAYING MULTIPLE GRAPHS  *********************")
    # Doing multiple graphs:
    graphs_in_experiment = list(data_structure.keys())
    graph_objects_list = [ data_structure[k]["example_graph"] for k in graphs_in_experiment[1:4]]

    return_value = graph_network(graph_objects_list,plot_type="2D",
                                    layout="random",
                                    colors=["blue","pink","purple"])
    
    
    
    
    #--------------------------------------------- how to do multiple graphs--------------------
    
    # Testing the directional in 3D:
    #creates the new directional graph
    # ------- MAKE THE FIRST GRAPH ------
    
    
    graphs_in_experiment = list(data_structure.keys())
    current_graph_example_1 = data_structure[graphs_in_experiment[2]]["example_graph"]
    current_graph_example_2 = data_structure[graphs_in_experiment[4]]["example_graph"]

    DG = nx.DiGraph()
    DG.add_edges_from(list(current_graph_example.edges))
    DG.add_nodes_from(list(current_graph_example.edges))
    #missing_nodes = [k for k in current_graph_example.nodes if k not in DG.nodes]


    #creates dictionary for node and edge colors
    node_colors = dict([(k,np.random.uniform(0,1,(3,))) for k in DG.nodes])

    edge_colors =  np.ones((len(DG.edges),3))
    edge_colors[:4] = np.array([0,1,0])
    edge_colors[4:7] = np.array([1,0,0])
    edge_colors[7:] = np.array([0,0,1])

    edge_colors_dict = dict([(k,edge_colors[i]) for i,k in enumerate(DG.edges)])

    # ------- MAKE THE SECOND GRAPH ------

    DG_2 = nx.DiGraph()
    DG_2.add_edges_from(list(current_graph_example_2.edges))
    DG_2.add_nodes_from(list(current_graph_example_2.edges))


    #creates dictionary for node and edge colors
    node_colors_2 = dict([(k,np.random.uniform(0,1,(3,))) for k in DG_2.nodes])

    edge_colors_2 =  np.ones((len(DG_2.edges),3))
    edge_colors_2[:4] = np.array([0,1,0])
    edge_colors_2[4:7] = np.array([1,0,0])
    edge_colors_2[7:] = np.array([0,0,1])

    edge_colors_dict_2 = dict([(k,edge_colors_2[i]) for i,k in enumerate(DG_2.edges)])


    graph_network([DG,DG_2],plot_type="3D",layout="random",locations=None,
                     #colors=[node_colors,node_colors_2],
                      colors=["blue","red"],
                     default_color="blue",
                      #colors_edge = [edge_colors_dict,edge_colors_dict_2],
                      colors_edge = ["pink","yellow"],
                      default_color_edge = "green",
                     plot_title="Example on Digraph",
                     return_fig = False)
                            
    """
    
    """   STEP 1
    1) Identify how many graphs have been passed and start loop that will go through all graphs
    - if multiply graphs then verify that plot type is all the same if it is a list
    """
    #make a copy of the graph so not alter attributes
    
    
    
    multiple_graph_flag = False
    if type(G) == list:
        multiple_graph_flag = True
        graph_list = [k.copy(as_view=False) for k in G]
    else: 
        if (type(G) != type(nx.Graph()) 
                and type(G) != type(nx.DiGraph())
                and type(G) != type(nx.MultiGraph())
                and type(G) != type(nx.MultiDiGraph())):
            raise Exception("The graph is not a network ")
        graph_list = [G.copy(as_view=False)]
            
    """         # ------------------------- WORKING ON LOCATION ---------------------------------- #
    3) Identify the locations vector specified: (identify as list or scalar --> turn to list)
    - if none in either override or graph then just save as empty array
    """
    
    def graph_locations(gr):
        extracted_location = nx.get_node_attributes(gr,"location")
        if len(extracted_location.keys()) <= 0:
            extracted_location = nx.get_node_attributes(gr,"locations")
        return extracted_location
    
    def dict_locations(gr,locations):
        if set(locations.keys()) == set(gr.nodes):
            return locations
        else:
            raise Exception("Dictionary passed for locations list does not match keys of graph")
            
    
    

    locations_list = []
    if type(locations) == list:
        if len(locations_list) != len(graph_list):
            raise Exception("locations list passed but does not match size of graph lists")
        for i,(l,gr) in enumerate(zip(locations,graph_list)):
            if type(l) == dict:
                locations_list.append(dict_locations(gr,l))
            else:
                locations_list.append(graph_locations(gr))
                
#             if set(l.keys()) == set(graph_list[i].nodes):
#                 current_locations = l
#             else:
#                 current_locations = dict() #stores empty dictionary if nothing or wrong thing provided
            
#             #append the current_locations to the total list
#             locations_list.append(current_locations)
        
    elif type(locations) == dict:
        for g in graph_list:
            locations_list.append(dict_locations(g,locations))
        
    else: #try to extract from the graph (will fill with empty list if not there)
        for i,gr in enumerate(graph_list):
            #print("Trying to extract locations from graph")
            locations_list.append(graph_locations(gr))
            
     
    #print("locations_list = " + str(locations_list))
    if len(locations_list) != len(graph_list):
            raise Exception("number of Location specified in list do not match number of graphs")
            


    """  # ------------------------- WORKING ON COLOR ---------------------------------- #
    2) check if color list specified: (identify as list or scalar --> turn to list)
    a. If yes --> make sure that it matches the number of nodes for all the graphs 
                (if "location" or "color" specified then create list for that)
    b. If no --> check for color attributes within the nodes of the graph
        i. if yes --> extract those as the color list
        ii. if no --> generate random color list for nodes
    
    """
    
    
    def string_color(colors,g,current_location=None):
        if colors == "locations":
            if set(g.nodes) != set(current_location.keys()):
                raise Exception("Color list specified as locations but the locations don't match the Graph list")
            return current_location
        else:
            return dict([(node_name,matplotlib.colors.to_rgb(colors)) for node_name in g.nodes()])
        
    def dict_color(colors,g):
        if set(g.nodes) != set(colors.keys()):
            raise Exception("Color list specified as dictionary not matching the nodes of the graph")
        else:
            return colors
    
    color_list = []
    if type(colors) == str:
        for g,current_location in zip(graph_list,locations_list):
            color_list.append(string_color(colors,g,current_location))
                      
    elif type(colors) == dict:
        for g in graph_list:
            color_list.append(dict_color(colors,g))
          
    elif type(colors) == list:
        if len(colors) != len(graph_list):
            raise Exception("Color list recieved but length of list not match length of graphs")
        for c,g,current_location in zip(colors,graph_list,locations_list):
            #print("Colors = " + str(c) + "type = " + str(type(c)))
            if type(c) == str:
                color_list.append(string_color(c,g,current_location))
            elif type(c) == dict:
                color_list.append(dict_color(c,g))
            else:
                raise Exception("Found color type in list that is not string or dictionary")
     
    else: #try to extract the colors from the graph (if not there then will just extract empty list)
        for i,gr in enumerate(graph_list):
            color_list.append(nx.get_node_attributes(graph_list[i],"colors"))
            
            
            
            
                
    #print(f"graph_list = {graph_list},\n location_list = {locations_list},\n color_list = {color_list}")
    
    
    
    
    if len(color_list) != len(graph_list):
         raise Exception("number of Colors specified in list do not match number of graphs")
            
    # go through and set all empty dictionaries to blue
    for i,c in enumerate(color_list):
        #print("c = " + str(c))
        if len(list(c.keys())) <= 0:
            print(f"Color list index {i} is blank so setting  color to default of {default_color}")
            color_list[i] = dict([(node_name,matplotlib.colors.to_rgb(default_color)) for node_name in graph_list[i].nodes])
            
    """  # ------------------------- WORKING ON EDGEEEEEE COLORING ---------------------------------- #
    1) build dictionary that maps the edges to the color
    
    """
    
    
    def string_color_edge(colors,g,current_location=None):
        #print("Colors = " + str(colors))
        #print("g = " + str(g))
        return dict([(node_name,matplotlib.colors.to_rgb(colors)) for node_name in g.edges()])
        
    def dict_color_edge(colors,g):
        if set(g.edges()) != set(colors.keys()):
            raise Exception("Color Edge list specified as dictionary not matching the nodes of the graph")
        else:
            return colors
    
    color_list_edge = []
    if type(colors_edge) == str:
        #print("Colors_edge = " + str(colors_edge))
        for g in graph_list:
            color_list_edge.append(string_color_edge(colors_edge,g))
                      
    elif type(colors_edge) == dict:
        for g in graph_list:
            color_list_edge.append(dict_color_edge(colors_edge,g))
          
    elif type(colors_edge) == list:
        if len(colors_edge) != len(graph_list):
            raise Exception("Color Edge list recieved but length of list not match length of graphs")
        for c,g in zip(colors_edge,graph_list):
            #print("Colors = " + str(c) + "type = " + str(type(c)))
            if type(c) == str:
                color_list_edge.append(string_color_edge(c,g,current_location))
            elif type(c) == dict:
                color_list_edge.append(dict_color_edge(c,g))
            else:
                raise Exception("Found color EDGE type in list that is not string or dictionary")
     
    else: #try to extract the colors from the graph (if not there then will just extract empty list)
        for i,gr in enumerate(graph_list):
            #print("No color edges provided")
            extracted_edge_colors = nx.get_edge_attributes(graph_list[i],"colors")
            if len(extracted_edge_colors) <= 0:
                extracted_edge_colors = nx.get_edge_attributes(graph_list[i],"color")
            color_list_edge.append(extracted_edge_colors)
            

    #print("color_list_edge = " + str(color_list_edge))
    if len(color_list_edge) != len(graph_list):
         raise Exception("number of Colors specified in list do not match number of graphs")
            
    # go through and set all empty dictionaries to blue
    for i,c in enumerate(color_list_edge):    
        if len(list(c.keys())) <= 0:
            print(f"Color list index EDGE {i} is blank so setting  color to default of {default_color_edge}")
            color_list_edge[i] = dict([(node_name,matplotlib.colors.to_rgb(default_color_edge)) for node_name in graph_list[i].edges])
            

    
    
    """# ------------------------- WORKING ON GRAPHING ---------------------------------- #"""
    
    from mpl_toolkits.mplot3d.proj3d import proj_transform
    from matplotlib.text import Annotation
    import matplotlib.pyplot as plt    
    from mpl_toolkits.mplot3d import axes3d
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    class Annotation3D(Annotation):
        '''Annotate the point xyz with text s'''

        def __init__(self, s, xyz, *args, **kwargs):
            Annotation.__init__(self,s, xy=(0,0), *args, **kwargs)
            self._verts3d = xyz        

        def draw(self, renderer):
            xs3d, ys3d, zs3d = self._verts3d
            xs, ys, zs = proj_transform(xs3d, ys3d, zs3d, renderer.M)
            self.xy=(xs,ys)
            Annotation.draw(self, renderer)

    def annotate3D(ax, s, *args, **kwargs):
        '''add anotation text s to to Axes3d ax'''

        tag = Annotation3D(s, *args, **kwargs)
        ax.add_artist(tag)


    #return locations_list, color_list
    if plot_type == "3D":
        fig = plt.figure()
        ax = fig.gca(projection='3d')
    else:
        fig = plt.figure()
    
    
    for i,(current_graph,current_locations,current_color,current_color_edge) in enumerate(zip(graph_list,locations_list,color_list,color_list_edge)):
        if plot_type == "3D":
            directional_flag = False
            if type(current_graph) == type(nx.DiGraph()) or type(G) == type(nx.MultiGraph()):
                directional_flag = True
                
            
            #check that there are valid locations
            location_check = set(current_locations.keys()) != set(current_graph.nodes)
            if location_check or layout=="random": #then use the random locations
                if location_check:
                    print(f"Using Random location because the current location keys did not match those of the nodes for graph {i}")
                else:
                    print(f"Using random location because random specified for graph {i}")
                variable_locations =np.array([[random.uniform(0, 1),
                           random.uniform(0, 1),
                           random.uniform(0, 1)] for i in range(0,len(current_graph.nodes))])

                current_locations = dict([(n,vl) for n,vl in zip(current_graph.nodes,variable_locations)])
            
            if set(current_locations.keys()) != set(current_graph.nodes):
                raise Exception(f"Specified 3D graph but length of locations keys do not match length of nodes in graph {i}")
                

            
            node_locations = np.array([current_locations[current_n] for current_n in current_graph.nodes])
            
            #print("node_locations = " + str(node_locations))
            #print("current_graph.edges = " + str(current_graph.edges))
            node_colors = np.array([current_color[current_n] for current_n in current_graph.nodes])
            node_colors_edge = np.array([current_color_edge[current_n] for current_n in current_graph.edges])
            
            #print("current_color_edge = " + str(repr(current_color_edge)))
            
            node_edges = np.array(list(current_graph.edges))
            
            
            """**** Start using matplotlib from here"""

        

            xyzn = node_locations
            edges = list(current_graph.edges())
            segments = [(xyzn[s], xyzn[t]) for s, t in edges]   
            node_labels = list(current_graph.nodes)

            # create figure        
           
            
            #ax.set_axis_off()

            # plot vertices
            ax.scatter(xyzn[:,0],xyzn[:,1],xyzn[:,2], marker='o', c = node_colors, s = node_size)    
            # plot edges
            edge_col = Line3DCollection(segments, lw=1,colors=node_colors_edge)
            ax.add_collection3d(edge_col)
            # add vertices annotation.
            for j, xyz_ in enumerate(xyzn): 
                annotate3D(ax, s=node_labels[j], xyz=xyz_, fontsize=node_label_size, xytext=(-3,3),
                           textcoords='offset points', ha='right',va='bottom') 

            ax.set_xlabel("X axis")
            ax.set_ylabel("Y axis")
            ax.set_zlabel("Z axis")
        else:
            """
            
            """
            
                #https://networkx.github.io/grave/latest/gallery/plot_grid.html
    
    
            def degree_colorer(node_attributes):
                color = node_attributes['color']
                shape = 'o' #random.choice(['s', 'o', '^', 'v', '8'])
                return {'color': color, 'size': 1000, 'shape': shape}

            def font_styler(attributes):
                return {'font_size': 8,
                        'font_weight': .5,
                        'font_color': 'k'}

            def pathological_edge_style(edge_attributes):
                color = edge_attributes['color']
                return {'color': color}

            
            network = current_graph
            #check if node color attributes already there
            nx.set_node_attributes(network, current_color, 'color')
            nx.set_edge_attributes(network, current_color_edge, 'color')
            
            #len(nx.get_node_attributes(current_graph,"color")) <= 0
            
            #print("current_color_edge = " + str(current_color_edge))
            location_check = set(current_locations.keys()) != set(current_graph.nodes)
            if location_check or layout=="random": #then use the random locations
                if location_check and layout!="random":
                    print(f"Using Random location because the current location keys did not match those of the nodes for graph {i}")
                else:
                    print(f"Using random location because random specified for graph {i}")
                plot_network(network,
                    node_style=degree_colorer,
                     edge_style=pathological_edge_style,
                     node_label_style=font_styler,
                     #edge_label_style=tiny_font_styler)
                    )
            else:
                if layout == "collapse_x":
                    #print("Collapsing along x axis for 2D visualziation")
                    plot_title += "\nCollapse along X axis"
                    plot_network(network,
                                 layout=lambda G: {node: (current_locations[node][1],
                                        current_locations[node][2]) for node in G},
                                node_style=degree_colorer,
                                node_label_style=font_styler,
                                edge_style=pathological_edge_style,
                     
                                )
                    plt.xlabel("Y axis")
                    plt.ylabel("Z axis")
                elif layout == "collapse_y":
                    #print("Collapsing along y axis for 2D visualziation")
                    plot_title += "\nCollapse along Y axis"
                    plot_network(network,
                                 layout=lambda G: {node: (current_locations[node][0],
                                        current_locations[node][2]) for node in G},
                                node_style=degree_colorer,
                                node_label_style=font_styler,
                                edge_style=pathological_edge_style,
                     
                                )
                    plt.xlabel("X axis")
                    plt.ylabel("Z axis")
                else:
                    #print("Collapsing along z axis for 2D visualziation")
                    if  layout != "collapse_z":
                        print("Plotting by collapsing acorss z because no valid layout was chosen")
                    plot_title += "\nCollapse along Z axis"
                    plot_network(network,
                                 layout=lambda G: {node: (current_locations[node][0],
                                        current_locations[node][1]) for node in G},
                                node_style=degree_colorer,
                                node_label_style=font_styler,
                                edge_style=pathological_edge_style,
                     
                                )
                    plt.xlabel("X axis")
                    plt.ylabel("Y axis")
                
                    
                    
    
    
    if return_fig:
        if plot_type == "2D":
            fig.suptitle(plot_title)
        else:
            pass
        return fig
    
    if plot_type == "3D":
        plt.show()
        
    else:
        fig.set_figheight(10)
        fig.set_figwidth(10)
        plt.show()    



def graph_network(G,plot_type="2D",layout="non_random",locations=None,
                 colors=None,
                 default_color="blue",
                  colors_edge = None,
                  default_color_edge = "green",
                 plot_title="",
                 return_fig = False):
    """
    Purpose: To graph a network in either 2D or 3D with 
    optional specifications for different colors and different layouts
    
    -------Parameters (all of these parameters can be lists so can support multiple graph): ----------
    G (nx.Graph) : The Graph that will be plotted
    plot_type (str) : Specifies whether you want a 2D or 3D graph
    layout (np.array of str): Either a word description of how the nodes should be layed out in 2D (random or projection)
                            Or cartesian/random in 3D
    
    *locations (dict) : Overrides the existing locations that may be store in the graph structure
                            as node attributes
                            
    *color_list (dict/str) : Specifies the colors of each node to be plotted 
                            (could be string and just set as "location" to have as default
                            or "color" name to specify colors for all)
                            
                            
    ------  Returns: None ---------
    
    Pseudocode: 
    1) Identify how many graphs have been passed and start loop that will go through all graphs
        - if multiply graphs then verify that plot type is all the same if it is a list
        
    2) check if color list specified: (identify as list or scalar --> turn to list)
        a. If yes --> make sure that it matches the number of nodes for all the graphs 
                    (if "locations" or "color" specified then create list for that)
        b. If no --> check for color attributes within the nodes of the graph
            i. if yes --> extract those as the color list
            ii. if no --> generate random color list for nodes
    3) Identify the locations vector specified: (identify as list or scalar --> turn to list)
        - if none in either override or graph then just save as empty array
    4) Identify the type of plot wanted (identify as list or scalar --> turn to list)
                ****** for 2 - 4, if a list was specified but not the same length as Graph 
                        (and not 1) then raise exception *******************
                        
    a. If 2D type: get the layout type:
        If random --> use grave with no color specifications (and specify colors)
        Others: 
        If projection --> use grave put specify the correct projection (and specify colors)
    b. If 3D type: 
        if random --> generate random locations and then plot using ipyvolume
        if blank or cartesian --> use the locations specified in graph or locations
        
    5) Continue for all graphs
    6) issue the plot command
        
        
        
    Example on how to use:
    
    # print("\n\n\n*************** SIMPLE GRAPH *********************")
    #Simple graph

    return_value = graph_network(current_graph_example,plot_type="3D",layout="random",
                                              colors="pink")



    #Specifying different colors and locations

    # How to specify new locations

    variable_locations =np.array([[random.uniform(0, 1),
                               random.uniform(0, 1),
                               random.uniform(0, 1)] for i in range(0,len(current_graph_example.nodes))])

    celii_locations = dict([(n,vl) for n,vl in zip(current_graph_example.nodes,variable_locations)])

    # How to specify new colors with dictionary
    print("\n\n\n*************** DICTATING THE COLORS AND DICTIONARIES  *********************")

    celii_colors= dict([(n,vl) for n,vl in zip(current_graph_example.nodes,variable_locations)])



    return_value = graph_network(current_graph_example,plot_type="3D",
                                                layout="collapse_z",
                                               #locations = celii_locations,
                                              colors="locations",
                                             #colors = celii_colors
                                                )


    print("\n\n\n*************** OVERLAYING MULTIPLE GRAPHS  *********************")
    # Doing multiple graphs:
    graphs_in_experiment = list(data_structure.keys())
    graph_objects_list = [ data_structure[k]["example_graph"] for k in graphs_in_experiment[1:4]]

    return_value = graph_network(graph_objects_list,plot_type="2D",
                                    layout="random",
                                    colors=["blue","pink","purple"])
    
    
    
    
    #--------------------------------------------- how to do multiple graphs--------------------
    
    # Testing the directional in 3D:
    #creates the new directional graph
    # ------- MAKE THE FIRST GRAPH ------
    
    
    graphs_in_experiment = list(data_structure.keys())
    current_graph_example_1 = data_structure[graphs_in_experiment[2]]["example_graph"]
    current_graph_example_2 = data_structure[graphs_in_experiment[4]]["example_graph"]

    DG = nx.DiGraph()
    DG.add_edges_from(list(current_graph_example.edges))
    DG.add_nodes_from(list(current_graph_example.edges))
    #missing_nodes = [k for k in current_graph_example.nodes if k not in DG.nodes]


    #creates dictionary for node and edge colors
    node_colors = dict([(k,np.random.uniform(0,1,(3,))) for k in DG.nodes])

    edge_colors =  np.ones((len(DG.edges),3))
    edge_colors[:4] = np.array([0,1,0])
    edge_colors[4:7] = np.array([1,0,0])
    edge_colors[7:] = np.array([0,0,1])

    edge_colors_dict = dict([(k,edge_colors[i]) for i,k in enumerate(DG.edges)])

    # ------- MAKE THE SECOND GRAPH ------

    DG_2 = nx.DiGraph()
    DG_2.add_edges_from(list(current_graph_example_2.edges))
    DG_2.add_nodes_from(list(current_graph_example_2.edges))


    #creates dictionary for node and edge colors
    node_colors_2 = dict([(k,np.random.uniform(0,1,(3,))) for k in DG_2.nodes])

    edge_colors_2 =  np.ones((len(DG_2.edges),3))
    edge_colors_2[:4] = np.array([0,1,0])
    edge_colors_2[4:7] = np.array([1,0,0])
    edge_colors_2[7:] = np.array([0,0,1])

    edge_colors_dict_2 = dict([(k,edge_colors_2[i]) for i,k in enumerate(DG_2.edges)])


    graph_network([DG,DG_2],plot_type="3D",layout="random",locations=None,
                     #colors=[node_colors,node_colors_2],
                      colors=["blue","red"],
                     default_color="blue",
                      #colors_edge = [edge_colors_dict,edge_colors_dict_2],
                      colors_edge = ["pink","yellow"],
                      default_color_edge = "green",
                     plot_title="Example on Digraph",
                     return_fig = False)
                            
    """
    
    """   STEP 1
    1) Identify how many graphs have been passed and start loop that will go through all graphs
    - if multiply graphs then verify that plot type is all the same if it is a list
    """
    #make a copy of the graph so not alter attributes
    
    
    
    multiple_graph_flag = False
    if type(G) == list:
        multiple_graph_flag = True
        graph_list = [k.copy(as_view=False) for k in G]
    else: 
        if (type(G) != type(nx.Graph()) 
                and type(G) != type(nx.DiGraph())
                and type(G) != type(nx.MultiGraph())
                and type(G) != type(nx.MultiDiGraph())):
            raise Exception("The graph is not a network ")
        graph_list = [G.copy(as_view=False)]
            
    """         # ------------------------- WORKING ON LOCATION ---------------------------------- #
    3) Identify the locations vector specified: (identify as list or scalar --> turn to list)
    - if none in either override or graph then just save as empty array
    """
    
    def graph_locations(gr):
        extracted_location = nx.get_node_attributes(gr,"location")
        if len(extracted_location.keys()) <= 0:
            extracted_location = nx.get_node_attributes(gr,"locations")
        return extracted_location
    
    def dict_locations(gr,locations):
        if set(locations.keys()) == set(gr.nodes):
            return locations
        else:
            raise Exception("Dictionary passed for locations list does not match keys of graph")
            
    
    

    locations_list = []
    if type(locations) == list:
        if len(locations_list) != len(graph_list):
            raise Exception("locations list passed but does not match size of graph lists")
        for i,(l,gr) in enumerate(zip(locations,graph_list)):
            if type(l) == dict:
                locations_list.append(dict_locations(gr,l))
            else:
                locations_list.append(graph_locations(gr))
                
#             if set(l.keys()) == set(graph_list[i].nodes):
#                 current_locations = l
#             else:
#                 current_locations = dict() #stores empty dictionary if nothing or wrong thing provided
            
#             #append the current_locations to the total list
#             locations_list.append(current_locations)
        
    elif type(locations) == dict:
        for g in graph_list:
            locations_list.append(dict_locations(g,locations))
        
    else: #try to extract from the graph (will fill with empty list if not there)
        for i,gr in enumerate(graph_list):
            #print("Trying to extract locations from graph")
            locations_list.append(graph_locations(gr))
            
     
    #print("locations_list = " + str(locations_list))
    if len(locations_list) != len(graph_list):
            raise Exception("number of Location specified in list do not match number of graphs")
            


    """  # ------------------------- WORKING ON COLOR ---------------------------------- #
    2) check if color list specified: (identify as list or scalar --> turn to list)
    a. If yes --> make sure that it matches the number of nodes for all the graphs 
                (if "location" or "color" specified then create list for that)
    b. If no --> check for color attributes within the nodes of the graph
        i. if yes --> extract those as the color list
        ii. if no --> generate random color list for nodes
    
    """
    
    
    def string_color(colors,g,current_location=None):
        if colors == "locations":
            if set(g.nodes) != set(current_location.keys()):
                raise Exception("Color list specified as locations but the locations don't match the Graph list")
            return current_location
        else:
            return dict([(node_name,matplotlib.colors.to_rgb(colors)) for node_name in g.nodes()])
        
    def dict_color(colors,g):
        if set(g.nodes) != set(colors.keys()):
            raise Exception("Color list specified as dictionary not matching the nodes of the graph")
        else:
            return colors
    
    color_list = []
    if type(colors) == str:
        for g,current_location in zip(graph_list,locations_list):
            color_list.append(string_color(colors,g,current_location))
                      
    elif type(colors) == dict:
        for g in graph_list:
            color_list.append(dict_color(colors,g))
          
    elif type(colors) == list:
        if len(colors) != len(graph_list):
            raise Exception("Color list recieved but length of list not match length of graphs")
        for c,g,current_location in zip(colors,graph_list,locations_list):
            #print("Colors = " + str(c) + "type = " + str(type(c)))
            if type(c) == str:
                color_list.append(string_color(c,g,current_location))
            elif type(c) == dict:
                color_list.append(dict_color(c,g))
            else:
                raise Exception("Found color type in list that is not string or dictionary")
     
    else: #try to extract the colors from the graph (if not there then will just extract empty list)
        for i,gr in enumerate(graph_list):
            color_list.append(nx.get_node_attributes(graph_list[i],"colors"))
            
            
            
            
                
    #print(f"graph_list = {graph_list},\n location_list = {locations_list},\n color_list = {color_list}")
    
    
    
    
    if len(color_list) != len(graph_list):
         raise Exception("number of Colors specified in list do not match number of graphs")
            
    # go through and set all empty dictionaries to blue
    for i,c in enumerate(color_list):
        #print("c = " + str(c))
        if len(list(c.keys())) <= 0:
            print(f"Color list index {i} is blank so setting  color to default of {default_color}")
            color_list[i] = dict([(node_name,matplotlib.colors.to_rgb(default_color)) for node_name in graph_list[i].nodes])
            
    """  # ------------------------- WORKING ON EDGEEEEEE COLORING ---------------------------------- #
    1) build dictionary that maps the edges to the color
    
    """
    
    
    def string_color_edge(colors,g,current_location=None):
        #print("Colors = " + str(colors))
        #print("g = " + str(g))
        return dict([(node_name,matplotlib.colors.to_rgb(colors)) for node_name in g.edges()])
        
    def dict_color_edge(colors,g):
        if set(g.edges()) != set(colors.keys()):
            raise Exception("Color Edge list specified as dictionary not matching the nodes of the graph")
        else:
            return colors
    
    color_list_edge = []
    if type(colors_edge) == str:
        #print("Colors_edge = " + str(colors_edge))
        for g in graph_list:
            color_list_edge.append(string_color_edge(colors_edge,g))
                      
    elif type(colors_edge) == dict:
        for g in graph_list:
            color_list_edge.append(dict_color_edge(colors_edge,g))
          
    elif type(colors_edge) == list:
        if len(colors_edge) != len(graph_list):
            raise Exception("Color Edge list recieved but length of list not match length of graphs")
        for c,g in zip(colors_edge,graph_list):
            #print("Colors = " + str(c) + "type = " + str(type(c)))
            if type(c) == str:
                color_list_edge.append(string_color_edge(c,g,current_location))
            elif type(c) == dict:
                color_list_edge.append(dict_color_edge(c,g))
            else:
                raise Exception("Found color EDGE type in list that is not string or dictionary")
     
    else: #try to extract the colors from the graph (if not there then will just extract empty list)
        for i,gr in enumerate(graph_list):
            #print("No color edges provided")
            extracted_edge_colors = nx.get_edge_attributes(graph_list[i],"colors")
            if len(extracted_edge_colors) <= 0:
                extracted_edge_colors = nx.get_edge_attributes(graph_list[i],"color")
            color_list_edge.append(extracted_edge_colors)
            

    #print("color_list_edge = " + str(color_list_edge))
    if len(color_list_edge) != len(graph_list):
         raise Exception("number of Colors specified in list do not match number of graphs")
            
    # go through and set all empty dictionaries to blue
    for i,c in enumerate(color_list_edge):    
        if len(list(c.keys())) <= 0:
            print(f"Color list index EDGE {i} is blank so setting  color to default of {default_color_edge}")
            color_list_edge[i] = dict([(node_name,matplotlib.colors.to_rgb(default_color_edge)) for node_name in graph_list[i].edges])
            

    
    
    """# ------------------------- WORKING ON GRAPHING ---------------------------------- #"""

    #return locations_list, color_list
    if plot_type == "3D":
        fig = ipv.figure()
    else:
        fig = plt.figure()
    
    
    for i,(current_graph,current_locations,current_color,current_color_edge) in enumerate(zip(graph_list,locations_list,color_list,color_list_edge)):
        if plot_type == "3D":
            directional_flag = False
            if type(current_graph) == type(nx.DiGraph()) or type(G) == type(nx.MultiGraph()):
                directional_flag = True
                
            
            #check that there are valid locations
            location_check = set(current_locations.keys()) != set(current_graph.nodes)
            if location_check or layout=="random": #then use the random locations
                if location_check:
                    print(f"Using Random location because the current location keys did not match those of the nodes for graph {i}")
                else:
                    print(f"Using random location because random specified for graph {i}")
                variable_locations =np.array([[random.uniform(0, 1),
                           random.uniform(0, 1),
                           random.uniform(0, 1)] for i in range(0,len(current_graph.nodes))])

                current_locations = dict([(n,vl) for n,vl in zip(current_graph.nodes,variable_locations)])
            
            if set(current_locations.keys()) != set(current_graph.nodes):
                raise Exception(f"Specified 3D graph but length of locations keys do not match length of nodes in graph {i}")
                

            
            node_locations = np.array([current_locations[current_n] for current_n in current_graph.nodes])
            
            #print("node_locations = " + str(node_locations))
            #print("current_graph.edges = " + str(current_graph.edges))
            node_colors = np.array([current_color[current_n] for current_n in current_graph.nodes])
            node_colors_edge = np.array([current_color_edge[current_n] for current_n in current_graph.edges])
            
            #print("current_color_edge = " + str(repr(current_color_edge)))
            
            node_edges = np.array(list(current_graph.edges))

            midpoints = []
            directions = []
            for n1,n2 in current_graph.edges:
                difference = node_locations[n1] - node_locations[n2]
                directions.append(difference)
                midpoints.append(node_locations[n2] + difference/2)
            directions = np.array(directions)
            midpoints = np.array(midpoints)
            
            rgb_edges_color = node_colors_edge
            #print("rgb_edges_color = " + str(repr(rgb_edges_color)))
            total_colors_options = np.unique(rgb_edges_color,axis=0)
            #print("total_colors_options = " + str(repr(total_colors_options)))
            for n in total_colors_options:
                #print("n = " + str(repr(n)))
                #print("np.where(rgb_edges_color == n)[0] = " + str(repr(np.where(rgb_edges_color == n)[0])))
                color_indices = np.where(np.sum(rgb_edges_color == n,axis=1) >= 3)[0]
                #color_indices = np.unique(np.where(rgb_edges_color == n)[0])
                #print(color_indices)


                ipv.plot_trisurf(node_locations[:,0], 
                                node_locations[:,1], 
                                node_locations[:,2], 
                                lines=node_edges[color_indices],
                                color = n#marker="sphere",
                                #size = 10)
                                             )
                if directional_flag:
                    current_midpoints = midpoints[color_indices]
                    current_directions = directions[color_indices]
                    ipv.pylab.quiver(current_midpoints[:,0],current_midpoints[:,1],current_midpoints[:,2],
                    current_directions[:,0],current_directions[:,1],current_directions[:,2],
                    size=2,
                    size_selected=20,
                    color = n)
            
#             edges_mesh = ipv.plot_trisurf(node_locations[:,0], 
#                             node_locations[:,1], 
#                             node_locations[:,2], 
#                             lines=noed_edges,
#                             color = default_color_edge)#marker="sphere",
#                             #size = 10)
            #edges_mesh.color = [0., 1., 0.]
            # how to plot things in 3D: 
            nodes_mesh = ipv.pylab.scatter(node_locations[:,0], 
                                    node_locations[:,1], 
                                    node_locations[:,2],
                                    color=node_colors,
                                    size = 2,
                                    
                                    marker = "sphere")

            #makes it plot in unit cube
            
                
            
            cube_limits = [-0.5,1.5]
            ipv.xlim(cube_limits[0],cube_limits[1])
            ipv.ylim(cube_limits[0],cube_limits[1])
            ipv.zlim(cube_limits[0],cube_limits[1])
        else:
            """
            
            """
            
                #https://networkx.github.io/grave/latest/gallery/plot_grid.html
    
    
            def degree_colorer(node_attributes):
                color = node_attributes['color']
                shape = 'o' #random.choice(['s', 'o', '^', 'v', '8'])
                return {'color': color, 'size': 1000, 'shape': shape}

            def font_styler(attributes):
                return {'font_size': 8,
                        'font_weight': .5,
                        'font_color': 'k'}

            def pathological_edge_style(edge_attributes):
                color = edge_attributes['color']
                return {'color': color}

            
            network = current_graph
            #check if node color attributes already there
            nx.set_node_attributes(network, current_color, 'color')
            nx.set_edge_attributes(network, current_color_edge, 'color')
            
            #len(nx.get_node_attributes(current_graph,"color")) <= 0
            
            #print("current_color_edge = " + str(current_color_edge))
            location_check = set(current_locations.keys()) != set(current_graph.nodes)
            if location_check or layout=="random": #then use the random locations
                if location_check and layout!="random":
                    print(f"Using Random location because the current location keys did not match those of the nodes for graph {i}")
                else:
                    print(f"Using random location because random specified for graph {i}")
                plot_network(network,
                    node_style=degree_colorer,
                     edge_style=pathological_edge_style,
                     node_label_style=font_styler,
                     #edge_label_style=tiny_font_styler)
                    )
            else:
                if layout == "collapse_x":
                    #print("Collapsing along x axis for 2D visualziation")
                    plot_title += "\nCollapse along X axis"
                    plot_network(network,
                                 layout=lambda G: {node: (current_locations[node][1],
                                        current_locations[node][2]) for node in G},
                                node_style=degree_colorer,
                                node_label_style=font_styler,
                                edge_style=pathological_edge_style,
                     
                                )
                    plt.xlabel("Y axis")
                    plt.ylabel("Z axis")
                elif layout == "collapse_y":
                    #print("Collapsing along y axis for 2D visualziation")
                    plot_title += "\nCollapse along Y axis"
                    plot_network(network,
                                 layout=lambda G: {node: (current_locations[node][0],
                                        current_locations[node][2]) for node in G},
                                node_style=degree_colorer,
                                node_label_style=font_styler,
                                edge_style=pathological_edge_style,
                     
                                )
                    plt.xlabel("X axis")
                    plt.ylabel("Z axis")
                else:
                    #print("Collapsing along z axis for 2D visualziation")
                    if  layout != "collapse_z":
                        print("Plotting by collapsing acorss z because no valid layout was chosen")
                    plot_title += "\nCollapse along Z axis"
                    plot_network(network,
                                 layout=lambda G: {node: (current_locations[node][0],
                                        current_locations[node][1]) for node in G},
                                node_style=degree_colorer,
                                node_label_style=font_styler,
                                edge_style=pathological_edge_style,
                     
                                )
                    plt.xlabel("X axis")
                    plt.ylabel("Y axis")
                
                    
                    
    
    if return_fig:
        if plot_type == "2D":
            fig.suptitle(plot_title)
        else:
            pass
        return fig
    
    if plot_type == "3D":
        ipv.show()
        
    else:
        fig.set_figheight(10)
        fig.set_figwidth(10)
        plt.show()    
        
