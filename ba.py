from conversion import eps_delta_to_zcdp, renyi_to_eps_delta, zcdp_to_eps_delta
import numpy as np
import networkx as nx
from graph import logW, priv_global, computeTwalk
from graphutils import gossip_matrix
import matplotlib.pyplot as plt
import matplotlib.cm as cm

G = nx.barabasi_albert_graph(200, 4)
G = nx.convert_node_labels_to_integers(G)

graph = gossip_matrix(G)
logG = logW(graph)
graph_ = nx.to_numpy_array(graph)



delta = 1e-6
epss = np.linspace(.001, 10, num = 7)
myepss = [eps_delta_to_zcdp(eps, delta) for eps in epss]


# Create a subplot with two columns: one for the graph and one for the curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot the graph on the left side (ax1)
pos = nx.spring_layout(G)
G.remove_edges_from(nx.selfloop_edges(G))

nx.draw(G, pos, ax=ax1, node_size=10)
ax1.set_title('Barabási–Albert Graph')
color_map = cm.get_cmap('viridis', len(myepss))



for i, eps in enumerate(myepss):
    color = color_map(i)
    alpha=2
    sigma =  2* alpha/eps
    T = computeTwalk(graph, sigma)
    cste, priv = priv_global(logG, T, alpha, sigma)
    values = np.mean(priv, axis=0)
    print(eps, np.mean(values))
    # Group the values by degree and compute the average
    degree_values = {}
    for node, value in zip(G.nodes, values):
        degree = G.degree(node)
        if degree not in degree_values:
            degree_values[degree] = []
        degree_values[degree].append(value)

    avg_values = {degree: sum(values) / len(values) for degree, values in degree_values.items()}

    # Plot the results
    degrees = list(avg_values.keys())
    avg_values = [avg_values[degree] for degree in degrees]
    avg_values = [zcdp_to_eps_delta(a, 1e-6) for a in avg_values]
    print(len(avg_values))
    plt.xscale('log')
    #plt.yscale('log')


    ax2.scatter(degrees, avg_values, label=f'eps = {epss[i]}', color=color)

ax2.set_xlabel('Degree')
ax2.set_ylabel('Average Privacy')
ax2.legend()
ax2.set_title('Average Privacy by Node Degree')

plt.tight_layout()
plt.savefig('ba.pdf')
plt.show()

T
