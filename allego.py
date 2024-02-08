import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from graph import gossip_matrix, logW, computeTwalk, priv_global, communicability
import matplotlib.cm

def process_ego(the_ego):
    name_edgelist = "facebook/"+str(the_ego)+".edges"
    my_graph = nx.read_edgelist(name_edgelist)
    my_graph = nx.relabel_nodes(my_graph, lambda x:int(x))
    Gcc = sorted(nx.connected_components(my_graph), key=len, reverse=True)
    G0 = my_graph.subgraph(Gcc[0]).copy()
    to_remove = []
    for node in G0.nodes():
        if G0.degree[node] <= 0:
            to_remove.append(node)
    for node in to_remove:
        G0.remove_node(node)
    n = G0.number_of_nodes()

    u = np.random.randint(n)
    G0 = nx.convert_node_labels_to_integers(G0, label_attribute="fb_id")

    graph = gossip_matrix(G0)

    alpha = 2
    sigma = 10

    c, priv = priv_global(logW(graph), computeTwalk(graph, sigma), alpha, sigma)
    G0.remove_edges_from(nx.selfloop_edges(G0))

    return G0, priv[1]






egos = [0, 107, 348, 414, 686, 698, 1684, 3437, 3980]

plt.figure(figsize=(10,10))
for i, ego in enumerate(egos):
    G0, colors = process_ego(ego)
    plt.subplot(3, 3, i+1)
    nx.draw(G0, node_color=colors, node_size=10, alpha=0.5, edge_color='xkcd:silver', width=.5, cmap=plt.cm.cividis)
plt.savefig("allego.pdf", bbox_inches='tight', pad_inches=0)
plt.show()

