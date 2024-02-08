import numpy as np
import networkx as nx
from scipy.linalg import eigh

def gossip_matrix(graph, debug=False, type_g="hamilton"):
	"""Computes W with appropriate weight for a matrix gossip from a given graph
	Parameters
	----------
	graph: networkx graph
	debug: verify that W is bistochastic and symmetric and print debug
	type_g: strategie to compute the weights
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
	

	if debug:
		A = np.array(nx.adjacency_matrix(graph).todense(), dtype=np.float)
		print(A)
		assert (A == A.T).all()
		for i in range(n):
			print(A[i].sum())
			assert np.isclose(A[i].sum(), 1)

	return graph


def contrib(Wt, u, w):
	"""Computes the sensibility toward u's input in the messages transmitted per w according to Wt
	Wt: matrix of communication n * n 
	u: node
	w: node
	Return the float
	"""
	base = np.linalg.norm(Wt[w])
	return (Wt[w][u]/base)**2


def compute_max_n_degree(graph, u):
	maxi = max(nx.degree(graph, nx.neighbors(graph, u)), key=lambda x: x[1])
	return maxi[1]

def degree_max(graph):
	maxi = max(nx.degree(graph), key=lambda x: x[1])
	return maxi[1]

def T_mix(graph, sigma):
	"""
	Compute the number of iterations needed to reaches the ball of the noise parametred by sigma in muffliato algorithm.
	It assume that 1/n sum_0 (x^0_v - x_mean)^2 is .25 (for instance mean .5 with 50% nodes at 0, 50% at 1)
	graph: networkx graph
	sigma: positive float
	Return integer T
	"""
	lambda_2 = vplambda(graph)
	return int(np.ceil(1/np.sqrt(lambda_2) * np.log(graph.number_of_nodes() / sigma**2 * max (.25, sigma**2))))

def T_mix_randomized(graph, sigma):
	"""
	Compute the number of iterations needed to reaches the ball of the noise parametred by sigma in muffliato algorithm.
	It assume that 1/n sum_0 (x^0_v - x_mean)^2 is .25 (for instance mean .5 with 50% nodes at 0, 50% at 1)
	Parameters
	----------
	graph: networkx graph
	sigma: positive float
	Returns
	-------
	T: int
	"""
	lambda_2 = vplambda(graph)
	return int(np.ceil(1/lambda_2 * np.log(graph.number_of_nodes() / sigma**2 * max (.25, sigma**2))))

def vplambda(graph):
	"""Compute the spectral gap of a networkx graph
	"""
	W = nx.to_numpy_array(graph)
	eigen = eigh(np.eye(graph.number_of_nodes())- W, eigvals_only=True, subset_by_index=[0, 1])
	lambda_2 = eigen[1]
	assert(0 < lambda_2 < 1)
	return lambda_2