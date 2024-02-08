import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from graph import gossip_matrix, logW, computeTwalk, priv_global, communicability
import matplotlib.cm

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams.update({'font.size': 16})


# ego list 3980 3437 1912 1684 698 414 348 107 0
the_ego = 0
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

features = dict()
name_feats="facebook/"+ str(the_ego)+".feat"
with open(name_feats) as f:
	for l in f.readlines():
		info_node = l.split()
		features[int(info_node[0])] = np.array(info_node[1:], dtype=np.int64)
print(G0.nodes[u]["fb_id"])
ref = features[G0.nodes[u]["fb_id"]]
print(ref.shape)


all_feats = np.zeros((n, ref.shape[0]))
for i in range(n):
	all_feats[i] = features[G0.nodes[i]["fb_id"]]


graph = gossip_matrix(G0)

alpha = 2
sigma = 10

c, priv = priv_global(logW(graph), computeTwalk(graph, sigma), alpha, sigma)

plt.figure()
n = graph.number_of_nodes()
T = n*n*np.log(n)
T_rism = computeTwalk(graph, sigma)
print("T RISM",T_rism, "n2logn",T)
print(np.log(T_rism)/n, "global comp")


plt.subplot(121)
plt.imshow(np.log(priv),cmap=plt.cm.cividis)
plt.colorbar(location="bottom")
plt.axis('off')

plt.title("Privacy loss")

com = communicability(graph)
print(np.max(com), "max com")
plt.subplot(122)
plt.imshow(np.log(com), cmap=plt.cm.cividis)
plt.colorbar(location="bottom")
plt.axis('off')

plt.title("Communicability")

plt.savefig("fbcom"+str(the_ego)+".pdf",bbox_inches='tight', pad_inches=0)

plt.figure()
G0.remove_edges_from(nx.selfloop_edges(G0))
nx.draw(G0, node_color=priv[1],node_size=30, alpha=0.5, edge_color='xkcd:silver', width=.2, cmap=plt.cm.cividis)

plt.savefig("fbgraph"+str(the_ego)+".pdf",bbox_inches='tight', pad_inches=0)


plt.show()
