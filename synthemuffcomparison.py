import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm

from random import seed
from tqdm import tqdm, trange

from muffliato import simulation, acceleratedsimulation
from graphutils import gossip_matrix, compute_max_n_degree, T_mix, degree_max, vplambda
from graph import computeTwalk, logW, priv_global, maxi_priv


# For passing automatic checks
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams.update({'font.size': 24})

# various constants
seed(0)
np.random.seed(0)
eps_local = 1
alpha = 2
sigma = np.sqrt(alpha / 2 * eps_local)


def summarize_vectors(list_vect):
    """Compute a summary of several trials by taking overall min, max, and the average
    Parameters
    ----------
    list_vec: list of numpy array
        List of the different trials statistics
    Returns
    -------
    summary: array, shape [max_length, 3]
        array with (mean, min, max) as function of the distance
    """
    print(list_vect, "listvect")
    max_length = 6
    print("maxi", max_length)
    complete = np.empty((len(list_vect), max_length, 3))
    summary= np.zeros((max_length, 3))
    complete[:] = np.NaN
    for i, l in enumerate(list_vect):
        complete[i][:len(l)] = l
    for i in range(max_length):
        summary[i][0] = np.nanmean(complete[:, i, 0])
        summary[i][1] = np.nanmin(complete[:, i, 1])
        summary[i][2] = np.nanmax(complete[:, i, 2])
    return summary


def to_interval(vector):
    """Utility function to convert (mean, min, max) into (mean, length of lower error, length of upper errors ) 
    """
    lower_error = vector[:, 0] - vector[:, 1]
    upper_error = vector[:, 2] - vector[:, 0]

    # Ensure no negative values in errors
    lower_error[lower_error < 0] = 0
    upper_error[upper_error < 0] = 0

    vector[:, 1] = lower_error
    vector[:, 2] = upper_error
    return vector

# Compute the privacy loss as function of the shortest path
def vector_loss(graph, u=0):
    """
    Compute an array of abscisse the distance to node u and with first coordinate mean privacy loss, the minimum privacy loss and maximum one
    Parameters
    ----------
    graph: networkx graph
    u: int
        the node from which distance are computed
    Returns
    -------
    stats: numpy array, shape (max dist, 3)
    """

    n = graph.number_of_nodes()

    print("Preprocessing of the graph")
    graph = gossip_matrix(graph)
    T = T_mix(graph, sigma)
    print("we need ",T, " iterations")
    print("Simulation of Muffliato")
    eps_inst, error, precision = acceleratedsimulation(graph, T, n, sigma=sigma, debug=False, approx=False, u=u)  
    print("Computing the privacy loss")
    distance = nx.shortest_path_length(graph, source=u)
    max_dist = max(distance.values())
    privacy_losses = [ [] for i in range(max_dist+1)]
    stats = np.zeros((max_dist, 3))
    eps_node = np.clip(eps_inst.sum(axis=0), 0, eps_local)
    for i in range(n):
        privacy_losses[distance[i]].append(eps_node[i])
    for i in range(max_dist):
        stats[i] = np.mean(privacy_losses[i]),  np.min(privacy_losses[i]), np.max(privacy_losses[i]) 
    return stats


# Compute the privacy loss as function of the shortest path 
# Compute the version under the assumption that the sender is unknown
def vector_loss_RW(graph, u=0):
    """
    Compute an array of abscisse the distance to node u and with first coordinate mean privacy loss, the minimum privacy loss and maximum one for the random walk
    Parameters
    ----------
    graph: networkx graph
    u: int
        the node from which distance are computed
    Returns
    -------
    stats: numpy array, shape (max dist, 3)
    """

    n = graph.number_of_nodes()

    print("Preprocessing of the graph")
    graph = gossip_matrix(graph)
    T = computeTwalk(graph, sigma)
    print("we need ",T, " iterations")
    print("Simulation of RW")
    priv = logW(graph)
    c, eps = priv_global(priv, T, alpha, sigma)

    print("Computing the privacy loss")
    distance = nx.shortest_path_length(graph, source=u)
    max_dist = max(distance.values())
    privacy_losses = [ [] for i in range(max_dist+1)]
    stats = np.zeros((max_dist, 3))
    eps[u][u] = eps_local # same arbitrary value for the node itself
    eps_node = np.clip(eps[u], 0, eps_local)
    for i in range(n):
        privacy_losses[distance[i]].append(eps_node[i])
    for i in range(max_dist):
        stats[i] = np.mean(privacy_losses[i]),  np.min(privacy_losses[i]), np.max(privacy_losses[i]) 
    return stats


# the version computes the privacy loss under the assumption that the sender is known
def vector_loss_RW(graph, u=0):
    """
    Compute an array of abscisse the distance to node u and with first coordinate mean privacy loss, the minimum privacy loss and maximum one for the random walk
    Parameters
    ----------
    graph: networkx graph
    u: int
        the node from which distance are computed
    Returns
    -------
    stats: numpy array, shape (max dist, 3)
    """

    n = graph.number_of_nodes()

    print("Preprocessing of the graph")
    graph = gossip_matrix(graph)
    T = computeTwalk(graph, sigma)
    print("we need ",T, " iterations")
    print("Simulation of RW")
    priv = logW(graph)
    maxi_p = maxi_priv(graph, priv)
    c, eps = priv_global(maxi_p, T, alpha, sigma)

    print("Computing the privacy loss")
    distance = nx.shortest_path_length(graph, source=u)
    max_dist = max(distance.values())
    privacy_losses = [ [] for i in range(max_dist+1)]
    stats = np.zeros((max_dist, 3))
    eps[u][u] = eps_local # same arbitrary value for the node itself
    eps_node = np.clip(eps[u], 0, eps_local)
    for i in range(n):
        privacy_losses[distance[i]].append(eps_node[i])
    for i in range(max_dist):
        stats[i] = np.mean(privacy_losses[i]),  np.min(privacy_losses[i]), np.max(privacy_losses[i]) 
    return stats

fig, ax = plt.subplots(figsize=(16,10))

right_side = ax.spines["right"]
right_side.set_visible(False)
up_side = ax.spines["top"]
up_side.set_visible(False)


ax.set_yscale('log', base=2)
ax.set_xlim([0, 25.5])
ax.set_ylim([1e-5, 1.2])
ax.axhline(y=1, label="LDP loss", color="xkcd:black", lw=3)

print("For the hypercube")
# For exponential graph
hypercube = nx.hypercube_graph(11)
hypercube = nx.convert_node_labels_to_integers(hypercube)
loss_exp = vector_loss(hypercube)
loss_exp = to_interval(loss_exp)
ax.plot([i for i in range(len(loss_exp))], loss_exp[:,0], marker='+', color="xkcd:royal blue", ls='--', lw=3, ms="20")

loss_exp_rw = vector_loss_RW(hypercube)
loss_exp_rw = to_interval(loss_exp_rw)
ax.plot([i for i in range(len(loss_exp_rw))], loss_exp_rw[:,0], label="Exponential", marker='+', color="xkcd:royal blue", lw=3, ms="20")

print("For the Erdos Renyi graph")
# For ER
n = 2048
trials = 5
connex = False
while not connex:	
	binomial = nx.gnp_random_graph(n, 1.2*np.log(n)/n)
	connex = nx.is_connected(binomial)
all_loss_ER = []
all_loss_ER_RW = []
for trial in range(trials):
    loss_er = vector_loss(binomial)
    all_loss_ER.append(loss_er)
    loss_er_RW = vector_loss_RW(binomial)
    all_loss_ER_RW.append(loss_er_RW)
loss_er = summarize_vectors(all_loss_ER)
loss_er = to_interval(loss_er)
loss_er_RW = summarize_vectors(all_loss_ER_RW)
loss_er_RW = to_interval(loss_er_RW)
ax.errorbar([i for i in range(len(loss_er))], loss_er[:,0], yerr= loss_er[:, 1:].T, color="xkcd:jungle green", capthick=1, capsize = 4, lw=3, ls='--' )
ax.errorbar([i for i in range(len(loss_er_RW))], loss_er_RW[:,0], yerr= loss_er_RW[:, 1:].T, label="Erdos Renyi", color="xkcd:jungle green", capthick=1, capsize = 4, lw=3 )


print("For geometric graph")
# For geometric
n = 2048
pos = {i: (np.random.random(), np.random.random()) for i in range(n)}
geometric = nx.random_geometric_graph(n, 0.07, pos=pos)
loss_geo = vector_loss(geometric)
loss_geo = to_interval(loss_geo)

plt.errorbar([i for i in range(len(loss_geo))], loss_geo[:,0], yerr=loss_geo[:, 1:].T, color='xkcd:tealish', capthick=1, capsize = 4, lw=3, ls="--")

loss_geo_rw = vector_loss_RW(geometric)
loss_geo_rw = to_interval(loss_geo_rw)

plt.errorbar([i for i in range(len(loss_geo_rw))], loss_geo_rw[:,0], label="Random Geometric", yerr=loss_geo_rw[:, 1:].T, color='xkcd:tealish', capthick=1, capsize = 4, lw=3)

print("For the grid")
# For grid
grid = nx.grid_2d_graph(45, 45)
grid = nx.convert_node_labels_to_integers(grid)
u = 1035
loss_grid = vector_loss(grid)
loss_grid = to_interval(loss_grid)

ax.errorbar([i for i in range(len(loss_grid))], loss_grid[:,0], yerr= loss_grid[:, 1:].T, color='xkcd:light blue', capthick=1, capsize = 4, lw=3 , ls="--")


loss_grid_rw = vector_loss_RW(grid)
loss_grid_rw = to_interval(loss_grid_rw)

ax.errorbar([i for i in range(len(loss_grid_rw))], loss_grid_rw[:,0], label="Grid", yerr= loss_grid_rw[:, 1:].T, color='xkcd:light blue', capthick=1, capsize = 4, lw=3 )

ax.set_xlabel("Shortest Path Length")
ax.set_ylabel("Privacy Loss")

lines = plt.gca().get_lines()
legend2 = plt.legend([lines[i] for i in [1,2]],['Gossip','Random walk'], loc="lower left")
plt.gca().add_artist(legend2)
ax.legend(loc='lower right')
fig.savefig("fig1bmaxicorr.pdf",  bbox_inches='tight', pad_inches=0)
plt.show()


