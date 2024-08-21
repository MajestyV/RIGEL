# This module use networkx to generate connection weight matrices.
# https://networkx.org/documentation/stable/index.html#

import numpy as np
import networkx as nx

def R_shuffle(node_number=0,path_length=0):
    x = [np.random.random() for i in range(path_length)]
    e = [int(i / sum(x) * (node_number-path_length)) + 1 for i in x]
    re = node_number - sum(e)
    u = [np.random.randint(0, path_length - 1) for i in range(re)]
    for i in range(re):
        e[u[i]] += 1
    return e

def Network_initial(network_name=None, network_size=300, density=0.2, Depth=10, MC_configure=None, random_seed=2048):
    rng = np.random.RandomState(random_seed)
    if network_name == "ER":
        rg = nx.erdos_renyi_graph(network_size, density, directed=False)  # ER
        R_initial = nx.adjacency_matrix(rg).toarray()
    elif network_name == "DCG":
        rg = nx.erdos_renyi_graph(network_size, density, directed=True)  # ER
        R_initial = nx.adjacency_matrix(rg).toarray()
    elif network_name == "DAG":
        if MC_configure is not None:
            xx = np.append(0, np.cumsum(MC_configure['number']))
            for i in range(xx.shape[0] - 1):
                Reject_index = 1
                for j in range(0, xx.shape[0] - 1):
                    if len(MC_configure[i + 1]) == np.sum(np.isin(MC_configure[i + 1], MC_configure[j + 1] + 1)):
                        Reject_index = 0
                if Reject_index == 1 and (MC_configure[i + 1] != 1).all():
                    print("fail to construct the DAN under current Memory commnity strcutrue configuration")
                    Reject_index = 2
            if Reject_index != 2:
                R_initial_0 = np.zeros((network_size, network_size))
                for i in range(xx.shape[0] - 1):
                    for j in range(xx.shape[0] - 1):
                        if len(MC_configure[i + 1]) == np.sum(np.isin(MC_configure[i + 1] + 1, MC_configure[j + 1])):
                            R_initial_0[xx[i]:xx[i + 1], xx[j]:xx[j + 1]] = 1
                R_initial = np.triu(R_initial_0, 1)
            else:
                R_initial = None

        else:
            xx = R_shuffle(network_size, Depth)
            # xx=np.array([3,4,3])
            # xx=np.array([60,60,60,60,60])
            # xx=np.array([30,30,30,30,30,30,30,30,30,30])*3
            rg = nx.complete_multipartite_graph(*tuple(xx))
            x = nx.adjacency_matrix(rg).toarray()
            R_initial = np.triu(x, 1)
            # R_initial= np.tril(x,1)
        Real_density = np.sum(R_initial > 0) * 1.0 / (network_size ** 2)
        if Real_density > 0 and density < Real_density:
            R_initial[rng.rand(*R_initial.shape) <= (1.0 - density / Real_density)] = 0
        R_initial = np.triu(R_initial, 1)
    return R_initial

