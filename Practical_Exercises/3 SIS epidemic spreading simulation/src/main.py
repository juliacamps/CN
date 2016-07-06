import networkx as nx
import os
import numpy as np
import matplotlib.pyplot as plt
import random as rd

plt.close("all")

# Create network
create = 1

# Options: 0-> ER ; 1-> CM
ops = [0,1]
# Netowork configurations
sizes = [500]
ks = [2,4,6,8]
exps = [2.2,2.5,2.7,3]
# SIS configuration parameters
Nrep = 10
p0 = 0.2
Tmax = 1000
Ttrans = 900
mus = [0.1, 0.5, 0.9]
step = 0.02
betas = np.arange(0, 1 + step, step)
# Ploting variables
colors = ['b-', 'm-', 'g-']
markers = ['o', 'o', 'o']
# Size
for N in sizes:
    # Network types
    for op in ops:
        if op == 0:
            confs = ks
        else:
            confs = exps
        # Configurations
        for conf in confs:
            # Loading/Creating the network
            # ER
            if op == 0:
                k = conf
                p = k / N
                network = "ER_N" + str(N) + "_exp" + str(k).replace(".","_")
                if create:
                    G = nx.erdos_renyi_graph(N, p, seed=None, directed=False)
                    G = nx.Graph(G)
                    G.remove_edges_from(G.selfloop_edges())
                    nx.write_pajek(G,os.getcwd()+r"\Networks_Pajek\NET_" + network + ".net")
                else:
                    G = nx.read_pajek(os.getcwd()+r"\Networks_Pajek\NET_" + network + ".net")
            # CM
            else:
                exp = conf
                network = "CM_N" + str(N) + "_exp" + str(exp).replace(".","_")
                if create:
                    degree_sequence = nx.utils.create_degree_sequence(N, nx.utils.powerlaw_sequence, exponent=exp)
                    if not sum(degree_sequence) % 2 == 0:
                        print("caution")
                        degree_sequence[1] += 1
                    G = nx.configuration_model(degree_sequence, create_using=None, seed=None)
                    G = nx.Graph(G)
                    G.remove_edges_from(G.selfloop_edges())
                    nx.write_pajek(G,os.getcwd()+r"\Networks_Pajek\NET_" + network + ".net")
                else:
                    G = nx.read_pajek(os.getcwd()+r"\Networks_Pajek\NET_" + network + ".net")

            # Prepare figure
            plt.figure()
            plt.title(network+", SIS (p0="+str(p0)+")")
            plt.ylabel("p")
            plt.xlabel("beta")

            graphics = []
            # For differnt mu
            mu_it = 0
            print("START: "+network)
            for mu in mus:
                graphic = []
                # For different beta
                for beta in betas:
                    # Initializing
                    for node in G.nodes():
                        rand = rd.uniform(0, 1)
                        if rand <= p0:
                            G[node]['infected'] = 1
                        else:
                            G[node]['infected'] = 0

                    # Averaging
                    results = []
                    # For each repetition
                    for i in range(1,Nrep+1):
                        # Before stationary state
                        for j in range(1,Ttrans+1):
                            infected = [x for x in G.nodes() if G[x]['infected'] == 1]
                            succeptible = [x for x in G.nodes() if G[x]['infected'] == 0]
                            infects = len(infected)
                            for it in infected:
                                # Spreading
                                neighbors = list(set(G.neighbors(it)) & set(succeptible))
                                for nei in neighbors:
                                    rand = rd.uniform(0, 1)
                                    if rand <= beta:
                                        G[nei]['infected'] = 1
                                        succeptible.remove(nei)
                                # Recovery
                                rand = rd.uniform(0, 1)
                                if rand <= mu:
                                    G[it]['infected'] = 0
                        result = []
                        # IN stationary state
                        for t in range(j,Tmax):
                            infected = [x for x in G.nodes() if G[x]['infected'] == 1]
                            succeptible = [x for x in G.nodes() if G[x]['infected'] == 0]
                            infects = len(infected)
                            for it in infected:
                                # Spreading
                                neighbors = list(set(G.neighbors(it)) & set(succeptible))
                                for nei in neighbors:
                                    rand = rd.uniform(0, 1)
                                    if rand <= beta:
                                        G[nei]['infected'] = 1
                                        succeptible.remove(nei)
                                        infects += 1
                                # Recovery
                                rand = rd.uniform(0, 1)
                                if rand <= mu:
                                    G[it]['infected'] = 0
                                    infects -= 1
                            fraction = infects/len(G)
                            result.append(fraction)
                        results.append(np.average(result, axis=None, weights=None, returned=False))
                    graphic.append(np.average(results, axis=None, weights=None, returned=False))
                # plot
                plt.plot(betas, graphic, colors[mu_it], marker=markers[mu_it], label="mu = "+str(mu))
                print(graphic)
                mu_it += 1
            plt.legend(loc=0)
            plt.savefig(os.getcwd()+r"\images\SIS_" + network + ".png")

# plt.show()