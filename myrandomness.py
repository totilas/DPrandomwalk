import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.spatial.distance import euclidean
from graph import logW, priv_global
from graphutils import gossip_matrix
import random
# Seed for reproducibility
np.random.seed(42)

# Number of nodes
n = 200

# Distance threshold within which nodes will be connected
radius = 0.2


def shuffle_node_values(G):
    # Extract the values of the nodes
    values = [G.nodes[node]['value'] for node in G.nodes]

    # Shuffle the values randomly
    np.random.shuffle(values)

    # Assign the shuffled values back to the nodes
    for node, value in zip(G.nodes, values):
        G.nodes[node]['value'] = value

    return G




T = 1500
alpha=2
sigma=1

# Generate a geometric random graph
G = nx.random_geometric_graph(n, radius)
pos = nx.get_node_attributes(G, 'pos')
# Add a unique value to each node that is the sum of its coordinates
values = []
for node, coordinates in nx.get_node_attributes(G, 'pos').items():
    value = sum(coordinates)
    G.nodes[node]['value'] = value
    values.append(value)

# Compute the distances between each pair of nodes
distances = []
for i in range(n):
    for j in range(i+1, n):  # Avoid computing distance with self and repeated pairs
        distance = euclidean(pos[i], pos[j])
        distances.append(((i, j), distance))

# Sort the pairs by increasing distance
sorted_distances = sorted(distances, key=lambda x: x[1])

# compute the gossip matrix for the graph
graph = gossip_matrix(G)
logG = logW(graph)
graph = nx.to_numpy_array(graph)

# Make a copy of the graph
G_copy = G.copy()

G_copy_bis = G.copy()
# Shuffle the values of the nodes in the graph G
G_copy_bis = shuffle_node_values(G_copy_bis)


n_iter = 200

# Define the random walk function
def random_walk(G, steps):
    # Start at the first node
    current_node = 0

    # Initialize variables to compute the running mean
    running_sum = G.nodes[current_node]['value']

    # Lists to store the values and the privacy of all nodes every n_iter steps 
    values_history = []
    privacy_history = []

    
    # Perform the random walk
    for step in range(steps):
        # Select a random neighbor
        next_node = random.choices(np.arange(graph.shape[0]), weights=graph[current_node])[0]

        # Update the running mean
        running_sum += G.nodes[next_node]['value']
        new_value = running_sum / (step + 1)
        G.nodes[next_node]['value'] = new_value

        # Move to the next node
        current_node = next_node

        # Save the values and the privacy loss for some iteration
        if step % n_iter == 0:
            values_history.append([G.nodes[node]['value'] for node in G.nodes])
            cste, priv = priv_global(logG, step, alpha, sigma)
            privacy_history.append(priv)


    return G, values_history, privacy_history




# Define a color-blind friendly colormap
cmap = cm.get_cmap('cividis')

# Create a plot with two subplots
fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

# Plot the original graph
# Remove self-loops from the graph G
G.remove_edges_from(nx.selfloop_edges(G))
nx.draw(G, pos, with_labels=False, node_size=40, node_color=values, cmap=cmap, edge_color='grey', alpha=0.5, ax=axs[0])
axs[0].set_title('Original Graph')

# Plot the copied graph after the random walk
G_copy_bis.remove_edges_from(nx.selfloop_edges(G_copy_bis))
values_copy = [G_copy_bis.nodes[node]['value'] for node in G_copy_bis.nodes]
nx.draw(G_copy_bis, pos, with_labels=False, node_size=40, node_color=values_copy, cmap=cmap, edge_color='grey', alpha=0.5, ax=axs[1])
axs[1].set_title('Shuffled graph')


# Perform a random walk of 100 steps on the copied graph
G_copy, values_history, privacy_history = random_walk(G_copy, T)

G_copy_bis, values_history_bis, privacy_history_bis = random_walk(G_copy_bis, T)


# Add a common colorbar
norm = plt.Normalize(min(values + values_copy), max(values + values_copy))
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
plt.colorbar(sm, ax=axs, label='Node Value')
plt.savefig('graphrandom.pdf')
plt.show()




# Define a function to compute the average values for each quantile
def compute_quantile(values, privacy, sorted_distance):
    # Divide the sorted_distances into 10 quantiles
    num_quantiles = 10
    quantile_size = len(sorted_distances) // num_quantiles

    avg_values = np.zeros(num_quantiles)
    avg_privacys = np.zeros(num_quantiles)

    current_quantile, i = 0, 0
    avg_value, avg_privacy = 0, 0
    
    for pair, _ in sorted_distances:
        avg_value += np.abs(values[pair[0]]- values[pair[1]])
        avg_privacy += privacy[pair[0]][pair[1]]
        i+=1
        if i == quantile_size:
            avg_values[int(current_quantile)] = avg_value/i
            avg_privacys[int(current_quantile)] = avg_privacy/i

            current_quantile += 1
            avg_value, avg_privacy = 0, 0
            i = 0
    return avg_values, avg_privacys

# Initialize lists to store the average values for each quantile
avg_values_history = []
avg_privacy_history = []

# Iterate through the values_history and privacy_history and compute the average values by quantile
for t in range(len(values_history)):
    # Average node values by quantile
    values_t = values_history[t]

    # Average privacy values by quantile
    privacy_t = privacy_history[t]

    avg_values_t, avg_privacy_t = compute_quantile(values_t, privacy_t, sorted_distances)

    avg_values_history.append(avg_values_t)
    avg_privacy_history.append(avg_privacy_t)



# Determine the number of time steps (T)
T = len(avg_values_history)

# Get a color map to generate shades of color
color_map = cm.get_cmap('viridis', T)

# Create a figure and axis for the values
fig, ax1 = plt.subplots()

# Plot the curves for avg_values_history (dashed)
for t in range(T):
    color = color_map(t)
    ax1.plot(avg_values_history[t], linestyle='-', color=color, label=t*n_iter)
plt.legend()
ax1.set_xlabel('Quantile')
ax1.set_ylabel('Average Values')
ax1.tick_params(axis='y')

# Create a second y-axis for the privacy
ax2 = ax1.twinx()
for t in range(T):
    color = color_map(t)
    ax2.plot(avg_privacy_history[t], color=color, ls='--')

ax2.set_ylabel('Average Privacy')
ax2.tick_params(axis='y')

# Add custom legends for Values and Privacy
#plt.legend(['Values (Dashed)', 'Privacy (Plain)'], loc='upper left')

plt.title('Average Values and Privacy by Quantiles')
plt.tight_layout()
plt.savefig('zgeo.pdf')

plt.show()



# Initialize lists to store the average values for each quantile
avg_values_history = []
avg_privacy_history = []

# Iterate through the values_history and privacy_history and compute the average values by quantile
for t in range(len(values_history_bis)):
    # Average node values by quantile
    values_t = values_history_bis[t]

    # Average privacy values by quantile
    privacy_t = privacy_history_bis[t]

    avg_values_t, avg_privacy_t = compute_quantile(values_t, privacy_t, sorted_distances)

    avg_values_history.append(avg_values_t)
    avg_privacy_history.append(avg_privacy_t)


# Create a figure and axis for the values
fig, ax1 = plt.subplots()

# Plot the curves for avg_values_history (dashed)
for t in range(T):
    color = color_map(t)
    ax1.plot(avg_values_history[t], linestyle='-', color=color, label=t*n_iter)
plt.legend()
ax1.set_xlabel('Quantile')
ax1.set_ylabel('Average Values')
ax1.tick_params(axis='y')

# Create a second y-axis for the privacy
ax2 = ax1.twinx()
for t in range(T):
    color = color_map(t)
    ax2.plot(avg_privacy_history[t], color=color, ls='--')

ax2.set_ylabel('Average Privacy')
ax2.tick_params(axis='y')

# Add custom legends for Values and Privacy
#plt.legend(['Values (Dashed)', 'Privacy (Plain)'], loc='upper left')

plt.title('Average Values and Privacy by Quantiles')
plt.tight_layout()
plt.savefig('ageo.pdf')

plt.show()

