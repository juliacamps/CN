import networkx as nx
import os
import numpy as np
import matplotlib.pyplot as plt
import random as rd
import statistics as st
import sys
sys.path.insert(0, 'lib')
from evaluator import castell_level, getTrialLevel
from read_data import Data
from history import History
from itertools import groupby

plt.close("all")

#Run settings
applyCustomDegreeAnalysis = True
applyCustomContentAnalysis = True
generateNetworkResults = True
generateGeneralResults = True
deploy_know_how_relations = True
deploy_rivality_relations = True
visualizeResutls = True
generateResults = generateNetworkResults or generateGeneralResults or applyCustomDegreeAnalysis or applyCustomContentAnalysis

#Experiment configurations
topCastells_considered = 2 
superiority_thr = 3 #Two crews must have a difference of at least 3 levels, when not, they are considered rivals
topTrials_considered = 5
minim_w_thr = 0.3 #Minimum weight of a relation to be mantained

#Analysis configurations
analysis_thresholds = [1,5,10]

#Definition of parameters
separator = "#"
season_sep = "TEMP_INI\n"
in_period = 1990
fi_period = 2015
seasons_evaluated = list(range(in_period,fi_period+1,1))

#Data paths
ref_path = os.getcwd()+r"\..\data\hist_"+str(in_period)+"_"+str(fi_period)+".txt"
images_path = os.getcwd()+r"\..\results\images\\"
pajek_path = os.getcwd()+r"\..\results\networks\\"

data = Data(separator=separator,ini=in_period,fin=fi_period,path=ref_path)
conf_name = "_conf_"
if deploy_know_how_relations:
    conf_name += "KH_"
if deploy_rivality_relations:
    conf_name += "RIV_"


#Loading Data
data_content = data.read_dataset(season_sep=season_sep)
data_model = History()
data_model.build(seasons_content=data_content,seasons_ids=seasons_evaluated)

#Building the model
temporal_network = {}
temporal_network_kh = {}
temporal_network_riv = {}
for i in range(0,len(seasons_evaluated),1):
    season = seasons_evaluated[i]
    network_name = r"Network_of_season_" + str(season)# + conf_name
    current_data = data_model.getSeason(season)
    G = nx.DiGraph(season=season)
    G_kh = nx.DiGraph(season=season)
    G_riv = nx.DiGraph(season=season)
    current_crews = current_data.getCrews()
    current_actuations = current_data.getActuations()
    for crew in current_crews.items():
        G.add_node(crew[0],crew=crew[1],color=[1,1,0])
        G_kh.add_node(crew[0], crew=crew[1], color=[1, 1, 0])
        G_riv.add_node(crew[0], crew=crew[1], color=[1, 1, 0])

    for item in current_actuations.items():
        actuation = item[1]
        participants = actuation.getParticipants()
        for participant_1 in participants:
            for participant_2 in participants:
                if participant_1 != participant_2:
                    is_rivality = False
                    if G.node[participant_2]['crew'].getLevel() - G.node[participant_1]['crew'].getLevel() >= superiority_thr:

                        if deploy_know_how_relations:

                            performance_level = max(list(set([castell_level(x) for x in actuation.getParticipation(participant_2)])))
                            topCastells = G.node[participant_1]['crew'].getTopCastells(topCastells_considered)

                            if performance_level >= topCastells[0]:

                                if G.has_edge(participant_1,participant_2):
                                    G[participant_1][participant_2]['weight'] += 1
                                    G_kh[participant_1][participant_2]['weight'] += 1
                                else:
                                    G.add_edge(participant_1, participant_2, weight=1, type='know_how', color='blue')
                                    G_kh.add_edge(participant_1, participant_2, weight=1, type='know_how', color='blue')
                    elif abs(G.node[participant_1]['crew']._level - G.node[participant_2]['crew']._level) < superiority_thr and deploy_rivality_relations:
                        performance_level_1 = max(list(set([castell_level(x) for x in actuation.getParticipation(participant_1)])))
                        performance_level_2 = max(list(set([castell_level(x) for x in actuation.getParticipation(participant_2)])))
                        topTrials_1 = G.node[participant_1]['crew'].getTopTrials(topTrials_considered)
                        topTrials_2 = G.node[participant_2]['crew'].getTopTrials(topTrials_considered)
                        if performance_level_1 >= topTrials_1[0] and performance_level_2 >= topTrials_2[0]:
                            trials_level_1 = max(list(
                                set([getTrialLevel(x) for x in actuation.getParticipation(participant_1)])))
                            trials_level_2 = max(list(
                                set([getTrialLevel(x) for x in actuation.getParticipation(participant_2)])))
                            if performance_level_1 >= performance_level_2 and trials_level_2 >= performance_level_1 or \
                                performance_level_2 > performance_level_1 and trials_level_1 >= performance_level_2:
                                if G.has_edge(participant_1, participant_2):
                                    G[participant_1][participant_2]['weight'] += 1
                                    G_riv[participant_1][participant_2]['weight'] += 1
                                else:
                                    G.add_edge(participant_1, participant_2, weight=1, type='rivality', color='red')
                                    G_riv.add_edge(participant_1, participant_2, weight=1, type='rivality', color='red')

#Temporal smoothing
    G_ant = None
    G_kh_ant = None
    G_riv_ant = None
    if season-1 in temporal_network:
        G_ant = temporal_network[season-1]
        G_kh_ant = temporal_network_kh[season - 1]
        G_riv_ant = temporal_network_riv[season - 1]

        #Operations for global model
        for u, v in G.edges():
            G[u][v]['weight'] *= 0.5
        for u, v in G_ant.edges():
            if G.has_edge(u,v):
                if G[u][v]['type'] == G_ant[u][v]['type']:
                    G[u][v]['weight'] += 0.5 * G_ant[u][v]['weight']
            elif G.has_node(u) and G.has_node(v):
                G.add_edge(u,v,weight=0.5*G_ant[u][v]['weight'],type=G_ant[u][v]['type'],color=G_ant[u][v]['color'])
        nodes_ant = G_ant.nodes()
        for node in G.nodes():
            if node in nodes_ant:
                G.node[node]['crew'].calculateGrowth(G_ant.node[node]['crew'])
                G.node[node]['crew'].calculateUnsafeness(G_ant.node[node]['crew'])
            else:
                G.node[node]['color'] = [0,1,0]
        for u, v in G.edges():
            if G[u][v]['weight'] < minim_w_thr:
                G.remove_edge(u,v)

        #Operations for know how model
        for u, v in G_kh.edges():
            G_kh[u][v]['weight'] *= 0.5
        for u, v in G_kh_ant.edges():
            if G_kh.has_edge(u, v):
                if G_kh[u][v]['type'] == G_kh_ant[u][v]['type']:
                    G_kh[u][v]['weight'] += 0.5 * G_kh_ant[u][v]['weight']
            elif G_kh.has_node(u) and G_kh.has_node(v):
                G_kh.add_edge(u, v, weight=0.5 * G_kh_ant[u][v]['weight'], type=G_kh_ant[u][v]['type'],
                           color=G_kh_ant[u][v]['color'])
        nodes_ant = G_kh_ant.nodes()
        for node in G_kh.nodes():
            if node in nodes_ant:
                G_kh.node[node]['crew'].calculateGrowth(G_kh_ant.node[node]['crew'])
                G_kh.node[node]['crew'].calculateUnsafeness(G_kh_ant.node[node]['crew'])
            else:
                G_kh.node[node]['color'] = [0, 1, 0]
        for u, v in G_kh.edges():
            if G_kh[u][v]['weight'] < minim_w_thr:
                G_kh.remove_edge(u, v)

        #Operations for rivality model
        for u, v in G_riv.edges():
            G_riv[u][v]['weight'] *= 0.5
        for u, v in G_riv_ant.edges():
            if G_riv.has_edge(u, v):
                if G_riv[u][v]['type'] == G_riv_ant[u][v]['type']:
                    G_riv[u][v]['weight'] += 0.5 * G_riv_ant[u][v]['weight']
            elif G_riv.has_node(u) and G_riv.has_node(v):
                G_riv.add_edge(u, v, weight=0.5 * G_riv_ant[u][v]['weight'], type=G_riv_ant[u][v]['type'],
                              color=G_riv_ant[u][v]['color'])
        nodes_ant = G_riv_ant.nodes()
        for node in G_riv.nodes():
            if node in nodes_ant:
                G_riv.node[node]['crew'].calculateGrowth(G_riv_ant.node[node]['crew'])
                G_riv.node[node]['crew'].calculateUnsafeness(G_riv_ant.node[node]['crew'])
            else:
                G_riv.node[node]['color'] = [0, 1, 0]
        for u, v in G_riv.edges():
            if G_riv[u][v]['weight'] < minim_w_thr:
                G_riv.remove_edge(u, v)

    #Plotting the results from the network
        if generateNetworkResults:
            plt.figure()
            plt.title(network_name.replace("_"," "))
            edges = G.edges()
            nodes = G.nodes()
            colors = [G[u][v]['color'] for u, v in edges]
            weights = [G[u][v]['weight']/2 for u, v in edges]
            growth_colors = [[1,min(1,max(0,1-(G.node[u]['crew'].getGrowth()/4))),0] if G.node[u]['color'][0] != 0 else G.node[u]['color'] for u in nodes]
            level_size = [round(4*(G.node[u]['crew'].getLevel()+1)) for u in nodes]
            nx.draw(G, node_size=level_size, node_color=growth_colors, arrows=True, edges=edges, edge_color=colors, width=weights, with_labels = True,  font_size=6)
            plt.savefig(images_path + network_name + ".png")
            nx.write_pajek(G, pajek_path + network_name + ".net")

# Save network as a temporal state
    temporal_network[season] = G
    temporal_network_kh[season] = G_kh
    temporal_network_riv[season] = G_riv

#Degree Analysis on the built temporal model
if generateGeneralResults:
    for j in range(0, 3, 1):
        if j == 0:
            relations = "ALL"
            data = [item[1] for item in temporal_network.items()]
        elif j == 1:
            relations = "KH"
            data = [item[1] for item in temporal_network_kh.items()]
        else:
            relations = "RIV"
            data = [item[1] for item in temporal_network_riv.items()]

        # Pulling data from the model
        degree_sequence = sorted([item+1 for G in data for item in G.out_degree(weight='weight').values()], reverse=False)

        # Experiment name
        analysis_name ="Degrees_Analysis_"+relations
        # degree_sequence = [x+1 for x in sorted(G.out_degree(weight='weight').values(), reverse=False)]  # degree sequence

        x = []
        y = []
        ycum = []
        [(x.append(g[0]), y.append(len(list(g[1])))) for g in groupby(degree_sequence)]
        y = [y1 / sum(y) for y1 in y]
        ycum = np.cumsum(y[::-1])[::-1]
        xmin = 1
        alpha = 1 + len(x) / sum([np.log((x1 / xmin) - 0.5) for x1 in x])
        print("MLE apha value obtained is: " + str(alpha))
        pwx = x
        pwy = [(alpha - 1) * xmin ** (alpha - 1) * x1 ** (-alpha) for x1 in x]

        ylog = [np.log10(y1) for y1 in ycum]
        xlog = [np.log10(x1) for x1 in x]

        A = np.vstack([xlog, np.ones(len(xlog))]).T
        gamma, c = np.linalg.lstsq(A, ylog)[0]
        print("Regressed value of gamma =" + str(gamma) + ", regressed value of C=" + str(c))
        yfit = [gamma * x1 + c for x1 in xlog]

        xexp = [10 ** x1 for x1 in xlog]
        yexp = [10 ** y1 for y1 in yfit]

        plt.figure()
        dmax = max(degree_sequence)
        dmin = min(degree_sequence)
        plt.title("Histogram of the degree distribution")
        plt.hist(degree_sequence, bins=10 ** np.linspace(np.log10(dmin), np.log10(dmax), 10))
        plt.xlabel("k in log scale")
        plt.ylabel("p(k) in log scale")
        plt.gca().set_xscale("log")
        plt.gca().set_yscale("log")
        plt.savefig(images_path + analysis_name +"_log_hist_dist_exponent.png")

        plt.figure()
        plt.title("Degree distribution plot")
        plt.ylabel("p(k)")
        plt.xlabel("k")
        plt.plot(x, y, 'b-', marker='o', label="True data")
        plt.plot(pwx, pwy, 'g-', label="MLE with resulting gamma'=" + str(alpha))
        plt.legend()
        plt.savefig(images_path + analysis_name +"_dist_exponent.png")

        plt.figure()
        plt.title("CCDF plot")
        plt.ylabel("p(k)")
        plt.xlabel("k")
        plt.plot(xexp, yexp, 'g-', label="Fitted power law function,\nwith gamma=" + str(gamma))
        plt.plot(x, ycum, 'b-', marker='o', label="True data")
        plt.legend()
        plt.savefig(images_path + analysis_name +"_CCDF_exponent.png")

        plt.figure()
        plt.title("Degree distribution log/log plot")
        plt.ylabel("p(k)")
        plt.xlabel("k")
        plt.loglog(x, y, 'b-', marker='o', label="True data")
        plt.plot(pwx, pwy, 'g-', label="MLE with resulting gamma'=" + str(alpha))
        plt.legend()
        plt.savefig(images_path + analysis_name +"_log_dist_exponent.png")

        plt.figure()
        plt.title("CCDF log/log plot")
        plt.ylabel("p(k)")
        plt.xlabel("k")
        plt.loglog(x, ycum, 'b-', marker='o', label="True data")
        plt.plot(xexp, yexp, 'g-', label="Fitted power law function,\nwith gamma=" + str(gamma))
        plt.legend()
        plt.savefig(images_path + analysis_name +"_log_CCDF_exponent.png")


#Custom Analysis on the built temporal model
if applyCustomDegreeAnalysis:

    # Ploting variables
    discretezation_pices = 10
    colors_1 = ['b-', 'c-', 'g-']
    colors_1_std = [[0.5,0.5,1],[0.5,1,1],[0.5,1,0.5]]
    colors_2 = ['y-', 'r-', 'm-']
    colors_2_std = [[1, 0.7, 0], [1, 0.5, 0.5], [1, 0.5, 1]]
    markers = ['o', '^']
    analysis_configurations = [{'crews':'Overall', 'period':'Overall (1991-2015)','season_ini':1991,'season_fin':2015,'level_ini':-1,'level_fin':44},
                               {'crews': 'Small', 'period':'Overall (1991-2015)','season_ini':1991,'season_fin':2015, 'level_ini': -1, 'level_fin': 14},
                               {'crews': 'Medium', 'period':'Overall (1991-2015)','season_ini':1991,'season_fin':2015, 'level_ini': 15, 'level_fin': 28},
                               {'crews': 'Large', 'period':'Overall (1991-2015)','season_ini':1991,'season_fin':2015, 'level_ini': 29, 'level_fin': 44},
                               {'crews':'Overall', 'period':'The first raise (1991-2002)','season_ini':1991,'season_fin':2002, 'level_ini':-1,'level_fin':44},
                               {'crews': 'Small', 'period':'The first raise (1991-2002)','season_ini':1991,'season_fin':2002, 'level_ini': -1, 'level_fin': 14},
                               {'crews': 'Medium', 'period':'The first raise (1991-2002)','season_ini':1991,'season_fin':2002, 'level_ini': 15, 'level_fin': 28},
                               {'crews': 'Large', 'period':'The first raise (1991-2002)','season_ini':1991,'season_fin':2002, 'level_ini': 29, 'level_fin': 44},
                               {'crews':'Overall', 'period':'The depression (2003-2009)','season_ini':2003,'season_fin':2009,'level_ini':-1,'level_fin':44},
                               {'crews':'Small', 'period':'The depression (2003-2009)','season_ini':2003,'season_fin':2009,'level_ini':-1,'level_fin':14},
                               {'crews': 'Medium', 'period': 'The depression (2003-2009)', 'season_ini': 2003, 'season_fin': 2009, 'level_ini': 15, 'level_fin': 28},
                               {'crews': 'Large', 'period': 'The depression (2003-2009)', 'season_ini': 2003, 'season_fin': 2009, 'level_ini': 29, 'level_fin': 44},
                               {'crews':'Overall', 'period':'The platinum period (2010-2015)','season_ini':2010,'season_fin':2015,'level_ini':-1,'level_fin':44},
                               {'crews':'Small', 'period':'The platinum period (2010-2015)','season_ini':2010,'season_fin':2015,'level_ini':-1,'level_fin':14},
                               {'crews': 'Medium', 'period':'The platinum period (2010-2015)','season_ini':2010,'season_fin':2015, 'level_ini': 15, 'level_fin': 28},
                               {'crews': 'Large', 'period':'The platinum period (2010-2015)','season_ini':2010,'season_fin':2015, 'level_ini': 29, 'level_fin': 44}]
    #For each configuration
    for conf in analysis_configurations:
        season_thr_ini = conf['season_ini']
        season_thr_fin = conf['season_fin']
        level_thr_ini = conf['level_ini']
        level_thr_end = conf['season_fin']

        # Naming the experiment
        conf_name = "Correlating_Degrees_Analysis_crews_" + conf['crews'] + "_period_" + (conf['period'][:conf['period'].find(" (")]).replace(" ","_")

        #Plot of the current configuration
        plt.figure()
        plt.title(conf_name,y=1.28)
        plt.ylabel("score")
        plt.xlabel("degree")

        #Relations
        for j in range(0,3,1):
            if j==0:
                relations = "ALL"
                data = [item[1] for item in temporal_network.items()  if item[0]>=season_thr_ini and item[0]<=season_thr_fin]
            elif j==1:
                relations = "KH"
                data = [item[1] for item in temporal_network_kh.items()  if item[0]>=season_thr_ini and item[0]<=season_thr_fin]
            else:
                relations = "RIV"
                data = [item[1] for item in temporal_network_riv.items()  if item[0]>=season_thr_ini and item[0]<=season_thr_fin]

            #Pulling data from the model
            growth_tuples = sorted([[item[1], G.node[item[0]]['crew'].getGrowth()] for G in data for item in
                                    G.out_degree(weight='weight').items() if G.node[item[0]]['crew'].getLevel()>=level_thr_ini and G.node[item[0]]['crew'].getLevel()<=level_thr_end], key=lambda tup: tup[0], reverse=False)
            unsafeness_tuples = sorted([[item[1], G.node[item[0]]['crew'].getUnsafeness()] for G in data for item in
                                      G.out_degree(weight='weight').items() if G.node[item[0]]['crew'].getLevel()>=level_thr_ini and G.node[item[0]]['crew'].getLevel()<=level_thr_end], key=lambda tup: tup[0], reverse=False)
            #discretization
            degree_sequence = [item[0] for item in growth_tuples]
            dmax = max(degree_sequence)
            dmin = min(degree_sequence)
            cut = (dmax-dmin)/float(discretezation_pices)
            degrees_discretized = [i*cut+cut/float(2) for i in range(0,discretezation_pices,1)]

            growth_discretized = []
            unsafeness_discretized = []
            for i in range(0,len(degrees_discretized),1):
                growth_discretized.append([item[1] for item in growth_tuples if item[0] > i*cut and item[0] < (i+1)*cut])
                unsafeness_discretized.append([item[1] for item in unsafeness_tuples if item[0] > i*cut and item[0] < (i+1)*cut])

            #Selection of velid data and transformations
            degrees_discretized = [degrees_discretized[i] for i in range(0,len(growth_discretized),1) if len(growth_discretized[i])>1]
            growth_discretized = [growth_discretized[i] for i in range(0, len(growth_discretized), 1) if len(growth_discretized[i]) > 1]
            unsafeness_discretized = [unsafeness_discretized[i] for i in range(0,len(unsafeness_discretized),1) if len(unsafeness_discretized[i])>1]

            growth_values_data = [[sum(items)/float(len(items)), st.pstdev(items)] for items in growth_discretized]
            unsafeness_values_data = [[sum(items)/float(len(items)), st.pstdev(items)] for items in unsafeness_discretized]
            growth_values = [item[0] for item in growth_values_data]
            unsafeness_values = [item[0] for item in unsafeness_values_data]
            growth_std = [item[1] for item in growth_values_data]
            unsafeness_std = [item[1] for item in unsafeness_values_data]

            #Representing the results from the experiments
            if len(degrees_discretized)>0 and len(growth_values)>0 and len(unsafeness_values)>0:
                x = degrees_discretized
                y = growth_values
                A = np.vstack([x, np.ones(len(x))]).T
                m, c = np.linalg.lstsq(A, y)[0]
                m = round(m, 2)
                plt.errorbar(degrees_discretized, growth_values, growth_std, color=colors_1_std[j] ,linestyle='None', marker='None')
                plt.plot(degrees_discretized, growth_values, colors_1[j], marker=markers[0], label="G ("+relations+") m="+str(m))
                x = degrees_discretized
                y = unsafeness_values
                A = np.vstack([x, np.ones(len(x))]).T
                m, c = np.linalg.lstsq(A, y)[0]
                m = round(m, 2)
                plt.errorbar(degrees_discretized, unsafeness_values, unsafeness_std, color=colors_2_std[j], linestyle='None', marker='None')
                plt.plot(degrees_discretized, unsafeness_values, colors_2[j], marker=markers[1], label="U ("+relations+") m="+str(m))

        plt.legend(bbox_to_anchor=(0., 1.02, 1., 0.502), loc=3, ncol=3, mode="expand", borderaxespad=0., fontsize='x-small')
        manager = plt.get_current_fig_manager()
        manager.resize(*manager.window.maxsize())
        plt.savefig(images_path+conf_name+".png", pad_inches=1)


#Extracting properties of the growing crews
if applyCustomContentAnalysis:
    number_of_crews_selected_per_season = 10
    conf_name_1 = "Analysis-Highest_Lowest-Growth"
    conf_name_2 = "Analysis-Highest_Lowest-Unsafeness"

    minim_interest_thr = 1
    discretezation_pices = 10
    colors_1 = ['b-', 'c-', 'g-']
    colors_1_std = [[0.5, 0.5, 1], [0.5, 1, 1], [0.5, 1, 0.5]]
    colors_2 = ['y-', 'r-', 'm-']
    colors_2_std = [[1, 0.7, 0], [1, 0.5, 0.5], [1, 0.5, 1]]
    markers = ['o', '*', '^']

    fig1 = plt.figure()
    # plt.title("New results Level-Unsafeness/Growth")
    plt.ylabel("score")
    plt.xlabel("growth")
    ax1 = fig1.add_subplot(111)

    fig2, (ax3, ax2) = plt.subplots(1, 2, sharey=True)
    # plt.title("Results related to Unsafeness")
    ax3.set_ylabel("score")
    ax3.set_xlabel("unsafeness")
    plt.xlabel("unsafeness")
    # ax2 = fig2.add_subplot(111)
    # ax3 = fig2.add_subplot(121)

    data = [item[1] for item in temporal_network.items()]

    #Obtain data from the nodes with higher indexes scores of growth and unsafeness
    growth_tuples_max = sorted([[item[1], G.graph['season'], G.node[item[0]]['crew'].getLevel(),
                                 G.node[item[0]]['crew'].getUnsafeness(), G.out_degree(item[0],weight='weigth'),
                                 G.in_degree(item[0],weight='weigth')] for G in data for item in
                                sorted([[node,G.node[node]['crew'].getGrowth()] for node in G.nodes()],
                                       key=lambda tup: tup[1], reverse=True)[:min(number_of_crews_selected_per_season,len(G.nodes()))]],
                               key=lambda tup: tup[0], reverse=False)
    unsafeness_tuples_max = sorted(
        [[item[1], G.graph['season'], G.node[item[0]]['crew'].getLevel(), G.node[item[0]]['crew'].getGrowth(), G.out_degree(item[0],weight='weigth'),
          G.in_degree(item[0],weight='weigth')] for G in data for item in sorted([[node, G.node[node]['crew'].getUnsafeness()] for node in G.nodes()],
        key=lambda tup: tup[1], reverse=True)[:min(number_of_crews_selected_per_season, len(G.nodes()))]], key=lambda tup: tup[0], reverse=False)

    # Obtain data from the nodes with lower indexes scores of growth and unsafeness
    growth_tuples_min = sorted([[item[1], G.graph['season'], G.node[item[0]]['crew'].getLevel(), G.node[item[0]]['crew'].getUnsafeness(),
          G.out_degree(item[0],weight='weigth'), G.in_degree(item[0],weight='weigth')] for G in data for item in
         sorted([[node, G.node[node]['crew'].getGrowth()] for node in G.nodes()], key=lambda tup: tup[1],
                reverse=False)[:min(number_of_crews_selected_per_season,len(G.nodes()))]], key=lambda tup: tup[0], reverse=False)

    unsafeness_tuples_min = sorted(
        [[item[1], G.graph['season'], G.node[item[0]]['crew'].getLevel(), G.node[item[0]]['crew'].getGrowth(), G.out_degree(item[0],weight='weigth'),
          G.in_degree(item[0],weight='weigth')] for G in data for item in sorted([[node, G.node[node]['crew'].getUnsafeness()] for node in G.nodes()],
                key=lambda tup: tup[1], reverse=False)[:min(number_of_crews_selected_per_season, len(G.nodes()))]],
        key=lambda tup: tup[0], reverse=False)


    # discretization
    # MAX
    growth_sequence = sorted([item[0] for item in growth_tuples_max], reverse=False)
    unsafeness_sequence = sorted([item[0] for item in unsafeness_tuples_max], reverse=False)

    # MIN
    growth_sequence_2 = sorted([item[0] for item in growth_tuples_min], reverse=False)
    unsafeness_sequence_2 = sorted([item[0] for item in unsafeness_tuples_min], reverse=False)

    gmax = max(growth_sequence)
    gmin = min(growth_sequence)
    umax = max(unsafeness_sequence)
    umin = min(unsafeness_sequence)

    gmax_2 = max(growth_sequence_2)
    gmin_2 = min(growth_sequence_2)
    umax_2 = max(unsafeness_sequence_2)
    umin_2 = min(unsafeness_sequence_2)

    gcut = (gmax - gmin) / float(discretezation_pices)
    ucut = (umax - umin) / float(discretezation_pices)

    gcut_2 = (gmax_2 - gmin_2) / float(discretezation_pices)
    ucut_2 = (umax_2 - umin_2) / float(discretezation_pices)

    # MAX
    growth_discretized_g = [gmin + i * gcut + gcut / float(2) for i in range(0, discretezation_pices, 1)]
    unsafeness_discretized_u = [umin + i * ucut + ucut / float(2) for i in range(0, discretezation_pices, 1)]
    # MIN
    growth_discretized_g_2 = [gmin_2 + i * gcut_2 + gcut_2 / float(2) for i in range(0, discretezation_pices, 1)]
    unsafeness_discretized_u_2 = [umin_2 + i * ucut_2 + ucut_2 / float(2) for i in range(0, discretezation_pices, 1)]

    level_discretized_g = []
    unsafeness_discretized_g = []
    out_degree_discretized_g = []
    level_discretized_g_2 = []
    unsafeness_discretized_g_2 = []
    out_degree_discretized_g_2 = []
    for i in range(0, len(growth_discretized_g), 1):
        # MAX
        level_discretized_g.append(
            [item[2] for item in growth_tuples_max if item[0] > i * gcut + gmin and item[0] < (i + 1) * gcut + gmin])
        unsafeness_discretized_g.append(
            [item[3] for item in growth_tuples_max if item[0] > i * gcut + gmin and item[0] < (i + 1) * gcut + gmin])
        out_degree_discretized_g.append(
        [item[4] for item in growth_tuples_max if item[0] > i * gcut + gmin and item[0] < (i + 1) * gcut + gmin])

    for i in range(0, len(growth_discretized_g_2), 1):
        # MIN
        level_discretized_g_2.append(
            [item[2] for item in growth_tuples_min if item[0] > i * gcut_2 + gmin_2 and item[0] < (i + 1) * gcut_2 + gmin_2])
        unsafeness_discretized_g_2.append(
            [item[3] for item in growth_tuples_min if item[0] > i * gcut_2 + gmin_2 and item[0] < (i + 1) * gcut_2 + gmin_2])
        out_degree_discretized_g_2.append(
            [item[4] for item in growth_tuples_min if item[0] > i * gcut_2 + gmin_2 and item[0] < (i + 1) * gcut_2 + gmin_2])

    level_discretized_u = []
    growth_discretized_u = []
    out_degree_discretized_u = []
    level_discretized_u_2 = []
    growth_discretized_u_2 = []
    out_degree_discretized_u_2 = []

    # MAX
    for i in range(0, len(unsafeness_discretized_u), 1):
        level_discretized_u.append(
            [item[2] for item in unsafeness_tuples_max if item[0] > i * ucut + umin and item[0] < (i + 1) * ucut + umin])
        growth_discretized_u.append(
            [item[3] for item in unsafeness_tuples_max if item[0] > i * ucut + umin and item[0] < (i + 1) * ucut + umin])
        out_degree_discretized_u.append(
            [item[4] for item in unsafeness_tuples_max if item[0] > i * ucut + umin and item[0] < (i + 1) * ucut + umin])

    # MIN
    for i in range(0, len(unsafeness_discretized_u_2), 1):
        level_discretized_u_2.append(
            [item[2] for item in unsafeness_tuples_min if item[0] > i * ucut_2 + umin_2 and item[0] < (i + 1) * ucut_2 + umin_2])
        growth_discretized_u_2.append(
            [item[3] for item in unsafeness_tuples_min if item[0] > i * ucut_2 + umin_2 and item[0] < (i + 1) * ucut_2 + umin_2])
        out_degree_discretized_u_2.append(
            [item[4] for item in unsafeness_tuples_min if item[0] > i * ucut_2 + umin_2 and item[0] < (i + 1) * ucut_2 + umin_2])

    # Selection of velid data and transformations
    # MAX
    growth_discretized_g = [growth_discretized_g[i] for i in range(0, len(level_discretized_g), 1) if
                           len(level_discretized_g[i]) > minim_interest_thr]
    unsafeness_discretized_u = [unsafeness_discretized_u[i] for i in range(0, len(level_discretized_u), 1) if
                            len(level_discretized_u[i]) > minim_interest_thr]

    level_discretized_g = [level_discretized_g[i] for i in range(0, len(level_discretized_g), 1) if
                           len(level_discretized_g[i]) > minim_interest_thr]
    level_discretized_u = [level_discretized_u[i] for i in range(0, len(level_discretized_u), 1) if
                          len(level_discretized_u[i]) > minim_interest_thr]

    out_degree_discretized_g = [out_degree_discretized_g[i] for i in range(0, len(out_degree_discretized_g), 1) if
                           len(out_degree_discretized_g[i]) > minim_interest_thr]
    out_degree_discretized_u = [out_degree_discretized_u[i] for i in range(0, len(out_degree_discretized_u), 1) if
                       len(out_degree_discretized_u[i]) > minim_interest_thr]

    unsafeness_discretized_g = [unsafeness_discretized_g[i] for i in range(0, len(unsafeness_discretized_g), 1) if
                              len(unsafeness_discretized_g[i]) > minim_interest_thr]
    growth_discretized_u = [growth_discretized_u[i] for i in range(0, len(growth_discretized_u), 1) if
                                len(growth_discretized_u[i]) > minim_interest_thr]
    # MIN
    growth_discretized_g_2 = [growth_discretized_g_2[i] for i in range(0, len(level_discretized_g_2), 1) if
                            len(level_discretized_g_2[i]) > minim_interest_thr]
    unsafeness_discretized_u_2 = [unsafeness_discretized_u_2[i] for i in range(0, len(level_discretized_u_2), 1) if
                                len(level_discretized_u_2[i]) > minim_interest_thr]

    level_discretized_g_2 = [level_discretized_g_2[i] for i in range(0, len(level_discretized_g_2), 1) if
                           len(level_discretized_g_2[i]) > minim_interest_thr]
    level_discretized_u_2 = [level_discretized_u_2[i] for i in range(0, len(level_discretized_u_2), 1) if
                           len(level_discretized_u_2[i]) > minim_interest_thr]

    out_degree_discretized_g_2 = [out_degree_discretized_g_2[i] for i in range(0, len(out_degree_discretized_g_2), 1) if
                                len(out_degree_discretized_g_2[i]) > minim_interest_thr]
    out_degree_discretized_u_2 = [out_degree_discretized_u_2[i] for i in range(0, len(out_degree_discretized_u_2), 1) if
                                len(out_degree_discretized_u_2[i]) > minim_interest_thr]

    unsafeness_discretized_g_2 = [unsafeness_discretized_g_2[i] for i in range(0, len(unsafeness_discretized_g_2), 1) if
                                len(unsafeness_discretized_g_2[i]) > minim_interest_thr]
    growth_discretized_u_2 = [growth_discretized_u_2[i] for i in range(0, len(growth_discretized_u_2), 1) if
                            len(growth_discretized_u_2[i]) > minim_interest_thr]

    #Growth operations
    # MAX
    level_values_data_g = [[sum(items) / float(len(items)), st.pstdev(items)] for items in level_discretized_g]
    unsafeness_values_data_g = [[sum(items) / float(len(items)), st.pstdev(items)] for items in
                              unsafeness_discretized_g]
    out_degree_values_data_g = [[sum(items) / float(len(items)), st.pstdev(items)] for items in
                                out_degree_discretized_g]

    level_values_g = [item[0] for item in level_values_data_g]
    unsafeness_values_g = [item[0] for item in unsafeness_values_data_g]
    out_degree_values_g = [item[0] for item in out_degree_values_data_g]
    level_std_g = [item[1] for item in level_values_data_g]
    unsafeness_std_g = [item[1] for item in unsafeness_values_data_g]
    out_degree_std_g = [item[1] for item in out_degree_values_data_g]

    # MIN
    level_values_data_g_2 = [[sum(items) / float(len(items)), st.pstdev(items)] for items in level_discretized_g_2]
    unsafeness_values_data_g_2 = [[sum(items) / float(len(items)), st.pstdev(items)] for items in
                                unsafeness_discretized_g_2]
    out_degree_values_data_g_2 = [[sum(items) / float(len(items)), st.pstdev(items)] for items in
                                out_degree_discretized_g_2]

    level_values_g_2 = [item[0] for item in level_values_data_g_2]
    unsafeness_values_g_2 = [item[0] for item in unsafeness_values_data_g_2]
    out_degree_values_g_2 = [item[0] for item in out_degree_values_data_g_2]
    level_std_g_2 = [item[1] for item in level_values_data_g_2]
    unsafeness_std_g_2 = [item[1] for item in unsafeness_values_data_g_2]
    out_degree_std_g_2 = [item[1] for item in out_degree_values_data_g_2]

    #Unsafeness operations
    # MAX
    level_values_data_u = [[sum(items) / float(len(items)), st.pstdev(items)] for items in level_discretized_u]
    growth_values_data_u = [[sum(items) / float(len(items)), st.pstdev(items)] for items in
                            growth_discretized_u]
    out_degree_values_data_u = [[sum(items) / float(len(items)), st.pstdev(items)] for items in
                                out_degree_discretized_u]
    level_values_u = [item[0] for item in level_values_data_u]
    growth_values_u = [item[0] for item in growth_values_data_u]
    out_degree_values_u = [item[0] for item in out_degree_values_data_u]
    level_std_u = [item[1] for item in level_values_data_u]
    growth_std_u = [item[1] for item in growth_values_data_u]
    out_degree_std_u = [item[1] for item in out_degree_values_data_u]

    # MIN
    level_values_data_u_2 = [[sum(items) / float(len(items)), st.pstdev(items)] for items in level_discretized_u_2]
    growth_values_data_u_2 = [[sum(items) / float(len(items)), st.pstdev(items)] for items in
                            growth_discretized_u_2]
    out_degree_values_data_u_2 = [[sum(items) / float(len(items)), st.pstdev(items)] for items in
                                out_degree_discretized_u_2]
    level_values_u_2 = [item[0] for item in level_values_data_u_2]
    growth_values_u_2 = [item[0] for item in growth_values_data_u_2]
    out_degree_values_u_2 = [item[0] for item in out_degree_values_data_u_2]
    level_std_u_2 = [item[1] for item in level_values_data_u_2]
    growth_std_u_2 = [item[1] for item in growth_values_data_u_2]
    out_degree_std_u_2 = [item[1] for item in out_degree_values_data_u_2]

    # Representing the results from the experiments of MAX GROWTH
    # Statistical results from second analysis: solving -> y = mx + c
    x = growth_discretized_g
    y = level_values_g
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y)[0]
    m = round(m, 2)
    avg = round(st.mean(y),1)
    ax1.errorbar(growth_discretized_g, level_values_g, level_std_g, color=colors_1_std[0], linestyle='None',
                 marker='None')
    ax1.plot(growth_discretized_g, level_values_g, colors_1[0], marker=markers[0], label="L (MAX), m="+str(m)+" ("+str(avg)+")")
    x = growth_discretized_g
    y = out_degree_values_g
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y)[0]
    m = round(m, 2)
    avg = round(st.mean(y),1)
    ax1.errorbar(growth_discretized_g, out_degree_values_g, out_degree_std_g, color=colors_1_std[1], linestyle='None',
                 marker='None')
    ax1.plot(growth_discretized_g, out_degree_values_g, colors_1[1], marker=markers[1], label="D (MAX), m="+str(m)+" ("+str(avg)+")")
    x = growth_discretized_g
    y = unsafeness_values_g
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y)[0]
    m = round(m, 2)
    avg = round(st.mean(y),1)
    ax1.errorbar(growth_discretized_g, unsafeness_values_g, unsafeness_std_g, color=colors_1_std[2],
                 linestyle='None',
                 marker='None')
    ax1.plot(growth_discretized_g, unsafeness_values_g, colors_1[2], marker=markers[2], label="U (MAX), m="+str(m)+" ("+str(avg)+")")

    # Representing the results from the experiments of MIN GROWTH
    x = growth_discretized_g_2
    y = level_values_g_2
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y)[0]
    m = round(m, 2)
    avg = round(st.mean(y),1)
    ax1.errorbar(growth_discretized_g_2, level_values_g_2, level_std_g_2, color=colors_2_std[0], linestyle='None',
                 marker='None')
    ax1.plot(growth_discretized_g_2, level_values_g_2, colors_2[0], marker=markers[0], label="L (MIN), m="+str(m)+" ("+str(avg)+")")
    x = growth_discretized_g_2
    y = out_degree_values_g_2
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y)[0]
    m = round(m, 2)
    avg = round(st.mean(y),1)
    ax1.errorbar(growth_discretized_g_2, out_degree_values_g_2, out_degree_std_g_2, color=colors_2_std[1],
                 linestyle='None',
                 marker='None')
    ax1.plot(growth_discretized_g_2, out_degree_values_g_2, colors_2[1], marker=markers[1], label="D (MIN), m="+str(m)+" ("+str(avg)+")")
    x = growth_discretized_g_2
    y = unsafeness_values_g_2
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y)[0]
    m = round(m, 2)
    avg = round(st.mean(y),1)
    ax1.errorbar(growth_discretized_g_2, unsafeness_values_g_2, unsafeness_std_g_2, color=colors_2_std[2],
                 linestyle='None',
                 marker='None')
    ax1.plot(growth_discretized_g_2, unsafeness_values_g_2, colors_2[2], marker=markers[2], label="U (MIN), m="+str(m)+" ("+str(avg)+")")

    # Representing the results from the experiments of MAX UNSAFENESS
    x = unsafeness_discretized_u
    y = level_values_u
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y)[0]
    m = round(m, 1)
    avg = round(st.mean(y))
    ax2.errorbar(unsafeness_discretized_u, level_values_u, level_std_u, color=colors_1_std[0], linestyle='None',
                 marker='None')
    ax2.plot(unsafeness_discretized_u, level_values_u, colors_1[0], marker=markers[0], label="L (MAX), m="+str(m)+" ("+str(avg)+")")
    x = unsafeness_discretized_u
    y = out_degree_values_u
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y)[0]
    m = round(m, 1)
    avg = round(st.mean(y))
    ax2.errorbar(unsafeness_discretized_u, out_degree_values_u, out_degree_std_u, color=colors_1_std[1], linestyle='None',
                 marker='None')
    ax2.plot(unsafeness_discretized_u, out_degree_values_u, colors_1[1], marker=markers[1], label="D (MAX), m="+str(m)+" ("+str(avg)+")")
    x = unsafeness_discretized_u
    y = growth_values_u
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y)[0]
    m = round(m, 1)
    avg = round(st.mean(y))
    ax2.errorbar(unsafeness_discretized_u, growth_values_u, growth_std_u, color=colors_1_std[2],
                 linestyle='None',
                 marker='None')
    ax2.plot(unsafeness_discretized_u, growth_values_u, colors_1[2], marker=markers[2],
             label="G (MAX), m="+str(m)+" ("+str(avg)+")")

    # Representing the results from the experiments of MIN UNSAFENESS
    x = unsafeness_discretized_u_2
    y = level_values_u_2
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y)[0]
    m = round(m, 1)
    avg = round(st.mean(y))
    ax3.errorbar(unsafeness_discretized_u_2, level_values_u_2, level_std_u_2, color=colors_2_std[0], linestyle='None',
                 marker='None')
    ax3.plot(unsafeness_discretized_u_2, level_values_u_2, colors_2[0], marker=markers[0], label="L (MIN), m="+str(m)+" ("+str(avg)+")")
    x = unsafeness_discretized_u_2
    y = out_degree_values_u_2
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y)[0]
    m = round(m, 1)
    avg = round(round(st.mean(y),1))
    ax3.errorbar(unsafeness_discretized_u_2, out_degree_values_u_2, out_degree_std_u_2, color=colors_2_std[1],
                 linestyle='None',
                 marker='None')
    ax3.plot(unsafeness_discretized_u_2, out_degree_values_u_2, colors_2[1], marker=markers[1], label="D (MIN), m="+str(m)+" ("+str(avg)+")")
    x = unsafeness_discretized_u_2
    y = growth_values_u_2
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y)[0]
    m = round(m, 1)
    avg = round(st.mean(y))
    ax3.errorbar(unsafeness_discretized_u_2, growth_values_u_2, growth_std_u_2, color=colors_2_std[2],
                 linestyle='None',
                 marker='None')
    ax3.plot(unsafeness_discretized_u_2, growth_values_u_2, colors_2[2], marker=markers[2],
             label="G (MIN) m="+str(m)+" ("+str(avg)+")")

    ax1.legend(bbox_to_anchor=(0., 1.02, 1., 0.502), loc=3, ncol=3, mode="expand", borderaxespad=0., fontsize='x-small')
    fig1.savefig(images_path + conf_name_1 + ".png")
    ax2.legend(bbox_to_anchor=(0., 1.02, 1., 0.502), loc=3, ncol=2, mode="expand", borderaxespad=0., fontsize='xx-small')
    ax3.legend(bbox_to_anchor=(0., 1.02, 1., 0.502), loc=3, ncol=2, mode="expand", borderaxespad=0., fontsize='xx-small')
    fig2.savefig(images_path + conf_name_2 + ".png")

#Visualize results instruction
if generateResults:
    plt.figure()
    if visualizeResutls:
        plt.show()

