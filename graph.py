import numpy as np
import networkx as nx
from scipy.linalg import eigh, inv
import matplotlib.pyplot as plt
from random import seed


def vplambda(graph):
	"""Compute the spectral gap of a networkx graph
	"""
	W = nx.to_numpy_array(graph)
	eigen = eigh(np.eye(graph.number_of_nodes())- W, eigvals_only=True, subset_by_index=[0, 1])
	lambda_2 = eigen[1]
	assert(0 < lambda_2 < 1)
	return lambda_2

def gossip_matrix(graph, type_g="hamilton"):
    """Computes W with appropriate weight for a matrix gossip from a given graph
    Parameters
    ----------
    graph: networkx graph
    type_g: strategy to compute the weights
    - regular: just an normalization
    - hamilton Wuv = 1 / (max du, duv - 1)
    - else Wuv = 1 / (du + dv - 1) 
    Return 
    ------
    graph: networkx graph
        graph modified with correct weights and self loop
    """
    if type_g == "regular":
        A = np.array(nx.adjacency_matrix(graph).todense(), dtype=np.float)
        for i in range(A.shape[0]):
            A[i] = A[i] / A[i].sum()
        graph = nx.from_numpy_array(A)

    elif type_g == "hamilton":
        degree = nx.degree(graph)
        for u in nx.nodes(graph):
            graph.add_edge(u,u)
        for u in nx.nodes(graph):
            out_w = 0
            for v in nx.neighbors(graph, u):
                if v != u:
                    w = 1 / (max(degree[u],degree[v]) - 1)
                    out_w += w
                    graph[u][v]['weight'] = w
            graph[u][u]['weight'] = 1 - out_w
    else : 
        degree = nx.degree(graph)
        for u in nx.nodes(graph):
            graph.add_edge(u,u)
        for u in nx.nodes(graph):
            out_w = 0
            for v in nx.neighbors(graph, u):
                if v != u:
                    w = 1 / (degree[u] + degree[v] - 1)
                    out_w += w
                    graph[u][v]['weight'] = w
            graph[u][u]['weight'] = 1 - out_w


    return graph

def logW(W):
    """
    Compute for a given gossip matrix the graph specific loss
    """
    W = nx.to_numpy_array(W)
    eigenvalues, eigenvectors = eigh(W, eigvals_only=False)
    l_eig = -np.log(1-eigenvalues[:-1])
    assert np.isclose(eigenvectors @ np.diag(eigenvalues) @eigenvectors.T , W).all()
    priv = eigenvectors[:,:-1] @ np.diag(l_eig) @ eigenvectors[:,:-1].T
    #np.fill_diagonal(priv, 0)

    #assert (priv.mean(axis=0)<=0).all() # the assumption
    return priv

def communicability(W):
    """
    Compute communicability of the graph. We do not use the networkx implem, because it returns a dict of dict instead of a matrix
    """
    W = nx.to_numpy_array(W)
    eigenvalues, eigenvectors = eigh(W, eigvals_only=False)
    com = eigenvectors @ np.diag(np.exp(eigenvalues)) @ eigenvectors.T
    np.fill_diagonal(com, 0)
    return com

def computeTwalk(graph, sigma):
    """
    Compute the number of steps for convergence of the RW in theory for a given the level of precision that should be achieved
    """
        
    lambda_2 = vplambda(graph)
    return int(20*np.ceil(1/lambda_2 * np.log(graph.number_of_nodes()))*(.25+sigma**2)/sigma**2 )


def priv_global(logW, T, alpha, sigma):
    n = logW.shape[0]
    constant = alpha*T *np.log(T)/(sigma**2 *n**2)
    priv = constant + alpha*T*logW/(sigma**2 *n)
    for i in range(n):
        priv[i][i]=0
    return constant, priv


def maxi_priv(graph, logW):
    # Initialize P with zeros
    P = np.zeros_like(logW)

    # Iterate over each node in the graph
    for u in graph.nodes():
        # Iterate over all other nodes v
        for v in range(len(logW)):
            # Initialize a variable to find the maximum
            max_value = - np.inf
            for w in graph.neighbors(v):
                # Ensure w' is not equal to u
                if w != v:
                    # Update the maximum value if necessary
                    max_value = max(max_value, logW[u][w])


            # Update P[u][v] with the maximum value found
            P[u][v] = max_value

    return P


if __name__ == "__main__":

    # various constants
    seed(0)
    np.random.seed(0)
    eps_local = 1
    alpha = 2
    sigma = np.sqrt(alpha / 2 * eps_local)
    print(sigma)
    hypercube = nx.hypercube_graph(7) #exponential graph
    regular = nx.random_regular_graph(3, 100) 
    d_cliques = nx.ring_of_cliques(5, 20) 
    sizes = [75, 75, 50]
    probs = [[0.25, 0.05, 0.02], [0.05, 0.35, 0.07], [0.02, 0.07, 0.40]]
    g = nx.stochastic_block_model(sizes, probs, seed=0) # communities with intra and inter link defines by prob list
    star = nx.star_graph(10)

    graph = gossip_matrix(hypercube)
    priv = logW(graph)
    print(np.max(priv), np.min(priv), np.mean(np.abs(priv)), np.std(np.abs(priv)), "max vertex dependant")
    T = computeTwalk(graph, sigma)
    n = graph.number_of_nodes()

    print("T", T, n*np.log(n))
    print((np.sum(priv, axis = 1)), np.sum(priv, axis = 1))
    c, eps = priv_global(priv, T, alpha, sigma)
    print("c", c)
    plt.subplot(121)
    plt.imshow(eps)
    plt.colorbar()
    plt.title("Privacy loss")

    com = communicability(graph)
    print(np.max(com), "max com")
    plt.subplot(122)
    plt.imshow(com)
    plt.colorbar()

    plt.title("Communicability")
    plt.savefig("/home/edwige/docs/these/rwgraphs/hoquet.png")



    plt.figure()
    graph.remove_edges_from(nx.selfloop_edges(graph))
    nx.draw(graph, node_color=priv[1],node_size=30, alpha=0.5, edge_color='xkcd:silver', width=.2, cmap=plt.cm.cividis)


    plt.savefig('/home/edwige/docs/these/rwgraphs/figi.png')
