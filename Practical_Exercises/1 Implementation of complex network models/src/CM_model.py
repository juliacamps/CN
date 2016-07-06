import networkx as nx
import os
import numpy as np
import matplotlib.pyplot as plt
import random as rd
from itertools import groupby

plt.close("all")

# Configuration Model
sizes = [200,1000,2000,3000]
exponents = [2.2, 2.5, 2.7, 3]
for N in sizes:
    for exponent in exponents:
        plt.close("all")
        degree_sequence=nx.utils.create_degree_sequence(N,nx.utils.powerlaw_sequence,exponent=exponent)
        if not sum(degree_sequence) % 2 == 0:
            raise nx.NetworkXError('Error: invalid sequence')
        G = nx.empty_graph(N)
        stublist = []
        for n in G:
            for i in range(degree_sequence[n]):
                stublist.append(n)
        rd.shuffle(stublist)
        while stublist:
            n1 = stublist.pop()
            n2 = stublist.pop()
            G.add_edge(n1, n2)
        G=nx.Graph(G)
        G.remove_edges_from(G.selfloop_edges())

        network = "CM"

        plt.figure()
        plt.title("Network plot N="+str(N)+" ("+network+" network, with exponent="+str(exponent)+")")
        nx.draw(G,node_size=20, node_color='b')
        plt.savefig(""+network+"_N" + str(N) + "_network_dist_exponent" + str(exponent).replace(".","_") + ".png")

        degree_sequence=sorted(nx.degree(G).values(),reverse=False) # degree sequence
        x = []
        y = []
        ycum = []
        [(x.append(g[0]), y.append(len(list(g[1])))) for g in groupby(degree_sequence)]
        y = [y1 / sum(y) for y1 in y]
        ycum = np.cumsum(y[::-1])[::-1]
        xmin = 1
        alpha = 1+len(x)/sum([np.log(x1/xmin-0.5) for x1 in x])
        print("MLE apha value obtained is: "+str(alpha))
        pwx = x
        pwy = [(alpha-1)*xmin**(alpha-1)*x1**(-alpha) for x1 in x]

        ylog = [np.log10(y1) for y1 in ycum]
        xlog = [np.log10(x1) for x1 in x]

        A = np.vstack([xlog, np.ones(len(xlog))]).T
        gamma, c = np.linalg.lstsq(A, ylog)[0]
        print("Regressed value of gamma ="+str(gamma)+", regressed value of C="+str(c))
        yfit = [gamma*x1+c for x1 in xlog]

        xexp = [10**x1 for x1 in xlog]
        yexp = [10**y1 for y1 in yfit]

        plt.figure()
        dmax = max(degree_sequence)
        dmin = min(degree_sequence)
        plt.title("Histogram of the degree distribution, "+network+" model m=" + str(exponent))
        plt.hist(degree_sequence, bins=10 ** np.linspace(np.log10(dmin), np.log10(dmax), 10))
        plt.xlabel("k in log scale")
        plt.ylabel("p(k) in log scale")
        plt.gca().set_xscale("log")
        plt.gca().set_yscale("log")
        plt.savefig(""+network+"_N" + str(N) + "_log_hist_dist_exponent" + str(exponent).replace(".","_") + ".png")

        plt.figure()
        plt.title("Degree distribution plot ("+network+" network, with exponent="+str(exponent)+")")
        plt.ylabel("p(k)")
        plt.xlabel("k")
        plt.plot(x,y,'b-',marker='o', label="True data")
        plt.plot(pwx,pwy,'g-', label="MLE with resulting gamma'="+str(alpha))
        plt.legend()
        plt.savefig(""+network+"_N" + str(N) + "_dist_exponent" + str(exponent).replace(".","_") + ".png")

        plt.figure()
        plt.title("CCDF plot ("+network+" network, with exponent="+str(exponent)+")")
        plt.ylabel("p(k)")
        plt.xlabel("k")
        plt.plot(xexp, yexp, 'g-', label="Fitted power law function,\nwith gamma=" + str(gamma))
        plt.plot(x,ycum,'b-',marker='o', label="True data")
        plt.legend()
        plt.savefig(""+network+"_N" + str(N) + "_CCDF_exponent" + str(exponent).replace(".","_") + ".png")

        plt.figure()
        plt.title("Degree distribution log/log plot ("+network+" network, with exponent="+str(exponent)+")")
        plt.ylabel("p(k)")
        plt.xlabel("k")
        plt.loglog(x,y,'b-',marker='o', label="True data")
        plt.plot(pwx,pwy,'g-',label="MLE with resulting gamma'="+str(alpha))
        plt.legend()
        plt.savefig(""+network+"_N" + str(N) + "_log_dist_exponent" + str(exponent).replace(".","_") + ".png")

        plt.figure()
        plt.title("CCDF log/log plot ("+network+" network, with exponent="+str(exponent)+")")
        plt.ylabel("p(k)")
        plt.xlabel("k")
        plt.loglog(x,ycum,'b-',marker='o', label="True data")
        plt.plot(xexp,yexp, 'g-', label="Fitted power law function,\nwith gamma="+str(gamma))
        plt.legend()
        plt.savefig(""+network+"_N"+str(N)+"_log_CCDF_exponent"+str(exponent).replace(".","_")+".png")

        nx.write_pajek(G,os.getcwd()+"graph_"+network+"_N"+str(N)+"_exp"+str(exponent).replace(".","_")+".txt")

# plt.show()



