import networkx as nx
import os
import numpy as np
import matplotlib.pyplot as plt
import random as rd
from itertools import groupby

plt.close("all")

# ER Model
sizes = [200,1000,2000,3000]
ks = [2, 4, 6, 8]
for N in sizes:
    for k in ks:
        plt.close("all")
        p = k/N
        G = nx.Graph()
        G.add_nodes_from(range(1,N+1))
        for n1 in G:
            for n2 in G:
                if n1 != n2 and not G.has_edge(n1,n2):
                    i = rd.uniform(0,1)
                    if i <= p:
                        G.add_edge(n1,n2)

        network = "ER"

        plt.figure()
        plt.title("Network plot (" + network + " network, with <k>=" + str(k) + ")")
        nx.draw(G, node_size=20, node_color='b')
        plt.savefig("" + network + "_N" + str(N) + "_network_dist_k" + str(k) + ".png")

        degree_sequence = sorted(nx.degree(G).values(), reverse=False)  # degree sequence
        x = []
        y = []
        ycum = []
        [(x.append(g[0]), y.append(len(list(g[1])))) for g in groupby(degree_sequence)]
        y = [y1 / sum(y) for y1 in y]
        ycum = np.cumsum(y[::-1])[::-1]
        xmin = 1
        # alpha = 1 + len(x) / sum([np.log(x1 / xmin - 0.5) for x1 in x])
        # print("MLE apha value obtained is: " + str(alpha))
        # pwx = x
        # pwy = [(alpha - 1) * xmin ** (alpha - 1) * x1 ** (-alpha) for x1 in x]

        # ylog = [np.log10(y1) for y1 in ycum]
        # xlog = [np.log10(x1) for x1 in x]

        # A = np.vstack([xlog, np.ones(len(xlog))]).T
        # gamma, c = np.linalg.lstsq(A, ylog)[0]
        # print("Regressed value of gamma =" + str(gamma) + ", regressed value of C=" + str(c))
        # yfit = [gamma * x1 + c for x1 in xlog]

        # xexp = [10 ** x1 for x1 in xlog]
        # yexp = [10 ** y1 for y1 in yfit]

        plt.figure()
        dmax = max(degree_sequence)
        dmin = min(degree_sequence)
        plt.title("Histogram of the degree distribution, " + network + " model m=" + str(k))
        plt.hist(degree_sequence, bins=10 ** np.linspace(np.log10(dmin), np.log10(dmax), 10))
        plt.xlabel("k in log scale")
        plt.ylabel("p(k) in log scale")
        plt.gca().set_xscale("log")
        plt.gca().set_yscale("log")
        plt.savefig("" + network + "_N" + str(N) + "_log_hist_dist_k" + str(k) + ".png")

        plt.figure()
        plt.title("Degree distribution plot (" + network + " network, with <k>=" + str(k) + ")")
        plt.ylabel("p(k)")
        plt.xlabel("k")
        plt.plot(x, y, 'b-', marker='o', label="True data")
        # plt.plot(pwx, pwy, 'g-', label="MLE with resulting gamma'=" + str(alpha))
        plt.legend()
        plt.savefig("" + network + "_N" + str(N) + "_dist_k" + str(k) + ".png")

        plt.figure()
        plt.title("CCDF plot (" + network + " network, with k=" + str(k) + ")")
        plt.ylabel("p(k)")
        plt.xlabel("k")
        # plt.plot(xexp, yexp, 'g-', label="Fitted power law function,\nwith gamma=" + str(gamma))
        plt.plot(x, ycum, 'b-', marker='o', label="True data")
        plt.legend()
        plt.savefig("" + network + "_N" + str(N) + "_CCDF_k" + str(k) + ".png")

        plt.figure()
        plt.title("Degree distribution log/log plot (" + network + " network, with k=" + str(k) + ")")
        plt.ylabel("p(k)")
        plt.xlabel("k")
        plt.loglog(x, y, 'b-', marker='o', label="True data")
        # plt.plot(pwx, pwy, 'g-', label="MLE with resulting gamma'=" + str(alpha))
        plt.legend()
        plt.savefig("" + network + "_N" + str(N) + "_log_dist_k" + str(k) + ".png")

        plt.figure()
        plt.title("CCDF log/log plot (" + network + " network, with <k>=" + str(k) + ")")
        plt.ylabel("p(k)")
        plt.xlabel("k")
        plt.loglog(x, ycum, 'b-', marker='o', label="True data")

        # plt.plot(xexp, yexp, 'g-', label="Fitted power law function,\nwith gamma=" + str(gamma))
        plt.legend()
        plt.savefig("" + network + "_N" + str(N) + "_log_CCDF_k" + str(k) + ".png")

        nx.write_pajek(G,os.getcwd()+"graph_"+network+"_N"+str(N)+"_exp"+str(k))

# plt.show()


