import networkx as nx
import os
import numpy as np
import matplotlib.pyplot as plt
import random as rd
from itertools import groupby
from bisect import bisect


plt.close("all")

# Barabasi-Albert
sizes = [200,1000,2000,3000]
ks = [2, 4, 6]
for N in sizes:
    for k in ks:
        plt.close("all")
        iniN = 3
        G = nx.Graph()
        G.add_nodes_from(range(1,iniN+1))
        for n1 in G:
            for n2 in G:
                if n1 != n2 and not G.has_edge(n1,n2):
                    G.add_edge(n1, n2)

        m = k/2
        p = k/N
        m1 = int(np.ceil(m))

        cumDegrees = [iniN-1] * iniN
        cumDegrees = (np.cumsum(cumDegrees)).tolist()

        for i in range(iniN,N+1):
            dmax = cumDegrees[-1]

            indexes = []

            for j in range(0, m1):

                index = rd.randint(0, dmax-1)
                index = bisect(cumDegrees,index)
                while index in indexes:
                    index = rd.randint(0, dmax - 1)
                    index = bisect(cumDegrees, index)
                indexes.append(index)

            indexes.sort(reverse=True)

            G.add_node(len(G))
            newN = G[len(G)-1]
            cumDegrees.append(cumDegrees[-1])
            s = 0


            for it in indexes:

                G.add_edge(it,len(G)-1)
                s = s+1
                if indexes[-1] != it:
                    last = indexes[indexes.index(it)+1]
                else:
                    last = len(cumDegrees)
                for ind in range(it,last):
                    cumDegrees[ind] = cumDegrees[ind]+s
                cumDegrees[last-1] += s


        network = "BA"

        plt.figure()
        plt.title("Network plot ("+network+" network, with <k>="+str(k)+")")
        nx.draw(G,node_size=20, node_color='b')
        plt.savefig(""+network+"_N" + str(N) + "_network_dist_k" + str(k) + ".png")

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

        xexp = [10 ** x1 for x1 in xlog]
        yexp = [10 ** y1 for y1 in yfit]

        plt.figure()
        dmax = max(degree_sequence)
        dmin = min(degree_sequence)
        plt.title("Histogram of the degree distribution, "+network+" model m=" + str(k))
        plt.hist(degree_sequence, bins=10 ** np.linspace(np.log10(dmin), np.log10(dmax), 10))
        plt.xlabel("k in log scale")
        plt.ylabel("p(k) in log scale")
        plt.gca().set_xscale("log")
        plt.gca().set_yscale("log")
        plt.savefig(""+network+"_N" + str(N) + "_log_hist_dist_k" + str(k) + ".png")

        plt.figure()
        plt.title("Degree distribution plot ("+network+" network, with <k>="+str(k)+")")
        plt.ylabel("p(k)")
        plt.xlabel("k")
        plt.plot(x,y,'b-',marker='o', label="True data")
        plt.plot(pwx,pwy,'g-', label="MLE with resulting gamma'="+str(alpha))
        plt.legend()
        plt.savefig(""+network+"_N" + str(N) + "_dist_k" + str(k) + ".png")

        plt.figure()
        plt.title("CCDF plot ("+network+" network, with k="+str(k)+")")
        plt.ylabel("p(k)")
        plt.xlabel("k")
        plt.plot(xexp, yexp, 'g-', label="Fitted power law function,\nwith gamma=" + str(gamma))
        plt.plot(x,ycum,'b-',marker='o', label="True data")
        plt.legend()
        plt.savefig(""+network+"_N" + str(N) + "_CCDF_k" + str(k) + ".png")

        plt.figure()
        plt.title("Degree distribution log/log plot ("+network+" network, with <k>="+str(k)+")")
        plt.ylabel("p(k)")
        plt.xlabel("k")
        plt.loglog(x,y,'b-',marker='o', label="True data")
        plt.plot(pwx,pwy,'g-',label="MLE with resulting gamma'="+str(alpha))
        plt.legend()
        plt.savefig(""+network+"_N" + str(N) + "_log_dist_k" + str(k) + ".png")

        plt.figure()
        plt.title("CCDF log/log plot ("+network+" network, with <k>="+str(k)+")")
        plt.ylabel("p(k)")
        plt.xlabel("k")
        plt.loglog(x,ycum,'b-',marker='o', label="True data")

        plt.plot(xexp,yexp, 'g-', label="Fitted power law function,\nwith gamma="+str(gamma))
        plt.legend()
        plt.savefig(""+network+"_N"+str(N)+"_log_CCDF_k"+str(k)+".png")

        nx.write_pajek(G,os.getcwd()+"graph_"+network+"_N"+str(N)+"_exp"+str(k))

# plt.show()






