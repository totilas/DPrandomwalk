import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from graph import gossip_matrix, logW, computeTwalk, priv_global, communicability
import matplotlib.cm

matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams.update({'font.size': 16})

graph = nx.davis_southern_women_graph()

np.random.seed(1)
graph = gossip_matrix(graph)

alpha = 2
sigma = 10

c, priv = priv_global(logW(graph), computeTwalk(graph, sigma), alpha, sigma)

plt.figure()
n = graph.number_of_nodes()
print("n", n)
T = n*n*np.log(n)
T_rism = computeTwalk(graph, sigma)
print("T RISM",T_rism, "n2logn",T)
print(np.log(T_rism)/n, "global comp")


plt.subplot(121)
plt.imshow(np.log(priv),cmap=plt.cm.cividis)
plt.axis('off')
plt.colorbar(location="bottom")
plt.title("Privacy loss")

com = communicability(graph)
print(np.max(com), "max com")
plt.subplot(122)
plt.axis('off')
plt.imshow(np.log(com)/np.log(2), cmap=plt.cm.cividis)
plt.colorbar(location="bottom")
plt.title("Communicability")

plt.savefig("south.pdf",bbox_inches='tight', pad_inches=0)

plt.figure()

plt.subplot(121)

graph.remove_edges_from(nx.selfloop_edges(graph))
pos = nx.spring_layout(graph)
nx.draw(graph, pos=pos,node_color=priv.sum(axis=0),node_size=50, alpha=1, edge_color='xkcd:silver', width=.2, cmap=plt.cm.cividis)
x_pos = 0.5 # Adjust the x position as needed
y_pos = 0. # Adjust the y position as needed
plt.text(x_pos, y_pos, "Mean Privacy Loss", ha='center', transform=plt.gca().transAxes)

plt.subplot(122)
centrality = nx.eigenvector_centrality(graph)
print(centrality)
centrality = np.array([v for v in centrality.values() ])
print(centrality)
nx.draw(graph, pos=pos,node_color=centrality,node_size=50, alpha=1, edge_color='xkcd:silver', width=.2, cmap=plt.cm.cividis)
plt.text(x_pos, y_pos, "Katz Centrality", ha='center', transform=plt.gca().transAxes)
plt.savefig("southgraph.pdf",bbox_inches='tight', pad_inches=0)

plt.show()