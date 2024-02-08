from copy import deepcopy
from graphutils import contrib, vplambda
import numpy as np
import networkx as nx
from tqdm import tqdm, trange


def gamma_tcheb(lambda2):
	"""Compute the factor gamma in the tchebychev polynomials for the acceleration
	Parameters
	----------
	lambda2: float,
		second smallest eigenvalue of the Laplacian
	Returns
	-------
	gamma: float
	"""
	return 2 * (1 - np.sqrt(lambda2 * (1 - lambda2/4))) / (1 - lambda2/2)**2

def simulation(graph, T, n, sigma=1, u=0, debug=False, approx=False, alpha=2):
	"""Run muffliato without Tchebychev acceleration
	Parameters
	----------
	graph: networkx graph
	T: int
		number of gossip steps
	n: int
		number of nodes
	sigma: float
	u: int
		fixed node for the computation of privacy
	debug: boolean, default: False
		print the gossip matrix
	approx: boulean, default: False
		compute the formula given in corollary 1
	gamma: float, default: None
		the gamma constant for Tchebychev polynomials, when not given it is computed as define in Algorithm 1
	alpha: float
		parameter of RDP
	Returns
	-------
	eps_inst: array, shape(T, nb of nodes)
		the privacy loss per step for each node
	error: array, shape (T,)
		the convergence of x towards the mean
	proba_2: array, shape (T, number of nodes)
		privacy loss when approx is used
	precision: float
		magnitude of the difference between the true and the noisy version. going below this precision is not meaningful
	"""
	np.random.seed(1)
	contribution = np.zeros(n) # array to stock the contribution of a neighbor
	eps_inst = np.zeros((T, n)) # privacy loss due to iteration t in node i
	error = np.zeros(T)  
	x_inst = np.zeros((T+1,n))
	x_exact = np.random.uniform(size=n)
	x_inst[0] = np.clip(x_exact + np.random.randn(n) * sigma, 0, 1)
	#x_exact = np.array([n]+[0]*(n-1))
	print("x", x_exact)
	#x_inst[0] = np.clip(x_exact + np.random.randn(n) * sigma, 0, 1)
	x_mean_exact = np.mean(x_exact)
	x_mean = np.mean(x_inst[0])
	precision = (x_mean - x_mean_exact)**2
	print("precision", precision)
	W = nx.to_numpy_array(graph)
	if debug:
		print("the gossip matrix", W)

	if approx:
		proba_2 = np.zeros((T, n))

	Wt = deepcopy(W)

	for t in trange(T):
		for v in nx.nodes(graph):
			contribution[v] = alpha * contrib(Wt, u, v) / (2 * sigma**2) 
			if approx:
				proba_2[t][v] = Wt[u][v]**2
		for v in nx.nodes(graph):
			for w in nx.neighbors(graph, v):
				if w != v:
					eps_inst[t][v] += contribution[w]
		x_inst[t+1] = W @ x_inst[t]
		error[t] = np.linalg.norm(x_inst[t]-x_mean)**2 /n
		Wt = W @ Wt
	if approx:
		return eps_inst, error, proba_2, precision
	return eps_inst, error, precision

def acceleratedsimulation(graph, T,n, sigma, u=0, debug=False, approx=False, gamma=None, alpha=2):
	"""Run muffliato 
	Parameters
	----------
	graph: networkx graph
	T: int
		number of gossip steps
	n: int
		number of nodes
	sigma: float
	u: int
		fixed node for the computation of privacy
	debug: boolean, default: False
		print the gossip matrix
	approx: boulean, default: False
		compute the formula given in corollary 1
	gamma: float, default: None
		the gamma constant for Tchebychev polynomials, when not given it is computed as define in Algorithm 1
	alpha: float
		parameter of RDP
	Returns
	-------
	eps_inst: array, shape(T, nb of nodes)
		the privacy loss per step for each node
	error: array, shape (T,)
		the convergence of x towards the mean
	proba_2: array, shape (T, number of nodes)
		privacy loss when approx is used
	precision: float
		magnitude of the difference between the true and the noisy version. going below this precision is not meaningful
	"""
	np.random.seed(0)
	if gamma is None:
		lambda2 = vplambda(graph)
		gamma = gamma_tcheb(lambda2)
		print("Gamma ", gamma, "for a lambda ", lambda2)
	contribution = np.zeros(n) # array to stock the contribution of a neighbor
	eps_inst = np.zeros((T, n)) # privacy loss due to iteration t in node i
	error = np.zeros(T) # 
	x_inst = np.zeros((T+1,n))
	x_exact = np.random.uniform(size=n)
	x_inst[0] = np.clip(x_exact + np.random.randn(n) * sigma, 0, 1)
	x_mean_exact = np.mean(x_exact)
	x_mean = np.mean(x_inst[0])
	precision = (x_mean - x_mean_exact)**2
	W = nx.to_numpy_array(graph)
	if debug:
		print("the gossip matrix", W)

	if approx:
		proba_2 = np.zeros((T, n))

	Wt = deepcopy(W)

	for t in trange(T):
		for v in nx.nodes(graph):
			contribution[v] = contrib(Wt, u, v)
			if approx:
				proba_2[t][v] = Wt[u][v]**2
		for v in nx.nodes(graph):
			for w in nx.neighbors(graph, v):
				if w != v:
					eps_inst[t][v] += contribution[w]
		if t == 0:
			x_inst[t+1] = W @ x_inst[t]
		else:
			x_inst[t+1] = gamma *  W @ x_inst[t] + (1 - gamma) * x_inst[t-1]

		error[t] = np.linalg.norm(x_inst[t]-x_mean)**2 /n
		Wt = W @ Wt
	if approx:
		return eps_inst, error, proba_2, precision
	return eps_inst, error, precision

def gossip_vector(theta_init, graph, T):
	"""
	Compute the result of a gossip for a vector theta instead of a single real
	Parameters
	----------
	theta_init : array, shape (n_nodes, p)
		The model parameters in each node at the beginning of the gossip
	graph: networkx graph
		Graph of commincation, with weights already computed
	T: int
		Number of gossip steps
	Returns
	-------
	theta: array, shape (n_n)
	"""
	n, p = theta_init.shape
	theta = deepcopy(theta_init)
	W = nx.to_numpy_array(graph)
	lambda2 = vplambda(graph)
	gamma = gamma_tcheb(lambda2)
	for idx in range(p):
		x_new, x_old = theta[:, idx], np.zeros(n)
		for t in range(T):
			x_old, x_new = x_new, gamma *  W @ x_new + (1 - gamma) * x_old	
		theta[:, idx] = x_new
	return theta