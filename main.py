import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms import bipartite
import random
from itertools import permutations
import math
import scipy.stats as st


def create_graph(num_agents, num_houses, prob, wt):
    """Creates a random (Erdos Renyi method) bipartite graph with disjoint sets of size `num_agents' and `num_houses'.
    The probability of an edge existing is described by `prob'. The weight of each edge is described by `wt'
    (0 for unweighted, 1 for borda weights and 2 for random weighted edges)."""
    G = bipartite.random_graph(num_agents, num_houses, prob)
    agents = {n for n, d in G.nodes(data=True) if d["bipartite"] == 0}
    houses = set(G) - agents
    if wt == 0:
        for e in G.edges():
            nx.set_edge_attributes(G, {(e[0], e[1]): {"weight": 1.0}})
    elif wt == 1:
        max_deg = max(G.degree(a) for a in agents)
        for a in agents:
            deg = G.degree(a)
            vals = random.sample(range(max_deg - deg + 1, max_deg + 1), deg)
            i = 0
            for h in G.edges(a):
                nx.set_edge_attributes(G, {(h[0], h[1]): {"weight": vals[i]}})
                i = i + 1
    else:
        for e in G.edges():
            nx.set_edge_attributes(G, {(e[0], e[1]): {"weight": random.randint(1, 100)}})
    return agents, houses, G


def matching_envy(G, agents, houses, c, tog):
    """Returns the Envy, i.e., number of envious agents, of a matching `c' between `agents' and `houses', on graph G.
    The parameter `tog' accounts for different matching formats of `c' (1 for edge lists, 0 for ordered list of
    houses)."""
    envy = 0
    if tog == 0:
        i = 0
        for a in agents:
            ha = c[i]
            if G.has_edge(a, ha):
                val = G[a][ha]["weight"]
            else:
                val = 0
            ea = 0
            for h in houses:
                if h in c and G.has_edge(a, h):
                    if G[a][h]["weight"] > val:
                        ea = 1
            envy = envy + ea
            i = i + 1
    else:
        for a, ha in c:
            if G.has_edge(a, ha):
                val = G[a][ha]["weight"]
            else:
                val = 0
            ea = 0
            for h in houses:
                if h in c and G.has_edge(a, h):
                    if G[a][h]["weight"] > val:
                        ea = 1
            envy = envy + ea
            for na in agents:
                if G.has_edge(na, ha):
                    diff = 0
                    for a1, h1 in c:
                        if a1 == na:
                            diff = max(G[na][ha]["weight"] - G[a1][h1]["weight"], 0)
                            break
                        else:
                            diff = -1
                    if diff == -1:
                        diff = G[na][ha]["weight"]
                    if diff > 0:
                        diff = 1
                    envy = envy + diff
    return envy


def matching_total_envy(G, agents, houses, c, tog):
    """Returns the Total Envy of a matching `c' between `agents' and `houses', on graph G.
    The parameter `tog' accounts for different matching formats of `c' (1 for edge lists, 0 for ordered list of
    houses)."""
    tot_envy = 0
    if tog == 0:
        i = 0
        for a in agents:
            ha = c[i]
            if G.has_edge(a, ha):
                val = G[a][ha]["weight"]
            else:
                val = 0
            ea = 0
            for h in houses:
                if h in c and G.has_edge(a, h):
                    if G[a][h]["weight"] > val:
                        ea = ea + G[a][h]["weight"] - val
            tot_envy = tot_envy + ea
            i = i + 1
    else:
        for a, ha in c:
            if G.has_edge(a, ha):
                val = G[a][ha]["weight"]
            else:
                val = 0
            ea = 0
            for h in houses:
                if h in c and G.has_edge(a, h):
                    if G[a][h]["weight"] > val:
                        ea = ea + G[a][h]["weight"] - val
            tot_envy = tot_envy + ea
            for na in agents:
                if G.has_edge(na, ha):
                    diff = 0
                    for a1, h1 in c:
                        if a1 == na:
                            diff = max(G[na][ha]["weight"] - G[na][h1]["weight"], 0)
                            break
                        else:
                            diff = -1
                    if diff == -1:
                        diff = G[na][ha]["weight"]
                    tot_envy = tot_envy + diff
    return tot_envy


def matching_welfare(G, agents, houses, m, tog):
    """Returns the Welfare, i.e., the sum of the weights of all edges, of an allocation `c' between `agents' and `houses',
    on graph G. The parameter `tog' accounts for different formats of `m'
    (1 for edge lists, 0 for ordered list of houses)."""
    welfare = 0
    if tog == 1:
        for a, h in m:
            if G.has_edge(a, h):
                welfare = welfare + G[a][h]["weight"]
    else:
        i = 0
        for a in agents:
            h = m[i]
            i = i + 1
            if G.has_edge(a, h):
                welfare = welfare + G[a][h]["weight"]
    return welfare


def mtemw(G, agents, houses):
    """Returns a Minimum Total Envy Maximum Welfare allocation between `agents' and `houses' on graph `G'."""
    mch = sorted(nx.max_weight_matching(G))
    wt = 0
    for a, h in mch:
        wt += G[a][h]["weight"]

    comb = list(permutations(houses, len(agents)))
    min_total_envy = math.inf
    ret_m = []
    for c in comb:
        a = 0
        welf = 0
        for i in c:
            if G.has_edge(a, i):
                welf += G[a][i]["weight"]
            a += 1
        total_envy = matching_total_envy(G, agents, houses, c, 0)
        if welf == wt and min_total_envy > total_envy:
            ret_m = c

    ret_match = []
    a = 0
    for i in ret_m:
        if G.has_edge(a, i):
            ret_match.append((a, i))
        a += 1
    return ret_match


def memw(G, agents, houses):
    """Returns a Minimum Envy Maximum Welfare allocation between `agents' and `houses' on graph `G'."""
    mch = sorted(nx.max_weight_matching(G))
    wt = 0
    for a, h in mch:
        wt += G[a][h]["weight"]

    comb = list(permutations(houses, len(agents)))
    min_envy = math.inf
    ret_m = []
    for c in comb:
        a = 0
        welf = 0
        for i in c:
            if G.has_edge(a, i):
                welf += G[a][i]["weight"]
            a += 1
        envy = matching_envy(G, agents, houses, c, 0)
        if welf == wt and min_envy > envy:
            ret_m = c

    ret_match = []
    a = 0
    for i in ret_m:
        if G.has_edge(a, i):
            ret_match.append((a, i))
        a += 1
    return ret_match


def match(G, agents, houses, type):
    """ Returns a `type' matching between `agents' and `houses' on `G'"""
    if type == 1:  # Minimum Envy Max Welfare
        ret_m = memw(G, agents, houses)
        return ret_m
    elif type == 2:  # Minimum Envy Complete and Minimum Total Envy Complete
        num_agents = len(agents)
        min_envy = math.inf
        min_total_envy = math.inf
        comb = list(permutations(houses, num_agents))
        ret_c_min_envy = comb[1]
        ret_c_min_total_envy = comb[1]
        for c in comb:
            envy = matching_envy(G, agents, houses, c, 0)
            total_envy = matching_total_envy(G, agents, houses, c, 0)
            if envy < min_envy:
                min_envy = envy
                ret_c_min_envy = c
            if total_envy < min_total_envy:
                min_total_envy = total_envy
                ret_c_min_total_envy = c
        return ret_c_min_envy, ret_c_min_total_envy, min_envy, min_total_envy
    else: #Minimum Total Envy Max Welfare
        ret_match = mtemw(G, agents, houses)
        return ret_match


def run_trials(num_agents, num_houses, prob, num_trials, wt):
    """Runs `num_trials' trials to find memw, mtemw, mec and mtec allocations on random house allocation instances
    with `num_agent' agents and `num_houses' houses, with `wt` (0 unweighted, 1 borda, 2 randomly weighted) edges,
      where each edge between agent a and house h exists with some probability p in `prob'"""
    trial_data = np.zeros((4, 3, len(num_houses), len(prob), num_trials))
    h_i = 0
    p_i = 0

    for i in num_houses:
        print("Number of houses : {}".format(i))
        for j in prob:
            print("Prob : {}".format(j))
            for k in range(num_trials):
                agents, houses, G = create_graph(num_agents, i, j, wt)
                bipartite.write_edgelist(G, "{}Houses/graph_{}Houses_{}prob_{}trial.txt".format(i, i, j, k + 1),
                                         data=True)
                # Min Envy Max Welfare
                m = match(G, agents, houses, 1)
                # Storing Data
                trial_data[0][0][h_i][p_i][k] = matching_envy(G, agents, houses, m, 1)
                trial_data[0][1][h_i][p_i][k] = matching_total_envy(G, agents, houses, m, 1)
                trial_data[0][2][h_i][p_i][k] = matching_welfare(G, agents, houses, m, 1)

                # Min Total Envy Max Welfare
                mt = match(G, agents, houses, 0)
                # Storing Data
                trial_data[3][0][h_i][p_i][k] = matching_envy(G, agents, houses, mt, 1)
                trial_data[3][1][h_i][p_i][k] = matching_total_envy(G, agents, houses, mt, 1)
                trial_data[3][2][h_i][p_i][k] = matching_welfare(G, agents, houses, mt, 1)

                # Min Envy/Total Envy Complete
                m1, m2, e1, te2 = match(G, agents, houses, 2)
                # Storing Data
                trial_data[1][1][h_i][p_i][k] = matching_total_envy(G, agents, houses, m1, 0)
                trial_data[2][0][h_i][p_i][k] = matching_envy(G, agents, houses, m2, 0)
                trial_data[1][2][h_i][p_i][k] = matching_welfare(G, agents, houses, m1, 0)
                trial_data[2][2][h_i][p_i][k] = matching_welfare(G, agents, houses, m2, 0)
                trial_data[1][0][h_i][p_i][k] = e1
                trial_data[2][1][h_i][p_i][k] = te2
            # Increase p_i
            p_i = p_i + 1
        # Move on to next house
        h_i = h_i + 1
        p_i = 0
    # End loops
    return trial_data


def get_y(data, prob, a, b, i):
    """Returns mean and the 95% CI of `data' for x values defined by `prob'. `a' and `b' define the appropriate entry in
    `data'."""
    y = []
    y_low = []
    y_high = []
    for j in range(len(prob)):
        d = data[a][b][i][j]
        d_mean = np.mean(d)
        c_i = st.t.interval(0.95, len(d) - 1,
                            loc=np.mean(d),
                            scale=st.sem(d))
        y_low.append(c_i[0])
        y_high.append(c_i[1])
        y.append(d_mean)
    return y, y_low, y_high


def plot_graphs(data, prob, wt, num_houses):
    """Plots 2D graphs representing y values stored in `data' and x values stored in `prob'."""
    x = prob
    # Envy
#     indices = [0, 2]
    indices = range(num_houses)
    stg = ["m = n", "", "m > n"]
    gs = [(0, "Envy"), (1, "Total Envy"), (2, "Welfare")]

    for t, g in gs:
        plt.figure()
        for i in indices:
            if g == "Total Envy" :
                y, y_low, y_high = get_y(data, prob, 3, t, i)
                lns = plt.plot(x, y, label="MTEMW; {}".format(stg[i]))
            else:
                y, y_low, y_high = get_y(data, prob, 0, t, i)
                if t == 0:
                    lns = plt.plot(x, y, label="MEMW; {}".format(stg[i]))
                else:
                    lns = plt.plot(x, y, label="MaxWelfare; {}".format(stg[i]))
            clr = lns[0].get_color()
            plt.fill_between(x, y_low, y_high, color=clr, alpha=.1)

            y, y_low, y_high = get_y(data, prob, 1, t, i)
            plt.plot(x, y, label="MEC; {}".format(stg[i]), linestyle='dashed', color=clr)
            plt.fill_between(x, y_low, y_high, color=clr, alpha=.1)

            y, y_low, y_high = get_y(data, prob, 2, t, i)
            plt.plot(x, y, label="MTEC; {}".format(stg[i]), linestyle='dotted', color=clr)
            plt.fill_between(x, y_low, y_high, color=clr, alpha=.1)

        plt.xlabel("Density of Graphs", fontsize=16)
        if t == 0:
            plt.ylabel("Number of Envious Agents", fontsize=16)
        elif t == 1:
            plt.ylabel("Total Envy", fontsize=16)
        else:
            plt.ylabel("Welfare", fontsize=16)

        if wt == 0:
            plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left', fontsize="small")
        else:
            plt.legend(loc='upper left', fontsize="small")
        plt.title("{} Graphs".format(g), fontsize=18)
        if t == 1:
            plt.savefig("TotalEnvyComb.png", bbox_inches='tight')
        else:
            plt.savefig("{}Comb.png".format(g), bbox_inches='tight')

def main():
    wt = 2  # 0 if binary/1 if borda/2 if weighted
    rerun = 0 # 0 if running trials from scratch / 1 if loading previous trial data

    num_agents = 5
    num_trials = 10
    num_houses = [5 , 8, 10]
    prob = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    if wt == 0:
        Mtype = "binary"
    elif wt == 1:
        Mtype = "borda"
    else:
        Mtype = "weighted"

    if rerun == 0:
        data = run_trials(num_agents, num_houses, prob, num_trials, wt)
        np.save("{}_agents_data_{}".format(num_agents, Mtype), data)
        np.savetxt("{}_houses_data_{}.txt".format(num_agents, Mtype), num_houses, delimiter=",")
        np.savetxt("{}_prob_data_{}.txt".format(num_agents, Mtype), prob, delimiter=",")
    else:
        filepath = "CI_{}/5_agents_data_{}.npy".format(Mtype, Mtype) #Enter filepath to previous trial data
        data = np.load(filepath)

    plot_graphs(data, prob, wt, len(num_houses))


if __name__ == "__main__":
    main()
