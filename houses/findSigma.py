import numpy as np
from scipy import optimize
from math import ceil


conf = 1.2

def loc(L, n_nodes, eps_tot, delta, n_iter):
	# max number of contributions
	K = ceil(conf *  n_iter / n_nodes)
	# delta should be divided by K as it sums up in compo
	delta_ = delta/K

	# solve associate max eps_0
	def f(eps_0):
		return np.sqrt(2*K*np.log(1/delta_))*eps_0 + K * eps_0 * (np.exp(eps_0)-1) - eps_tot

	eps_0 = optimize.bisect(f, 0, eps_tot/np.sqrt(K))
	print("   The per-step value of eps is", eps_0)

	# deduce sigma
	sigma = np.sqrt(2 *np.log(1.25/delta_))*L/eps_0
	print("   so we deduce that the sigma for Local DPSGD is", sigma)
	return sigma


def dpsgd(L, n_nodes, eps_tot, delta, n_iter):
	# Bassily et al. bound
	sigma = 16*L *np.sqrt(n_iter * np.log(2/delta)*np.log(1.25 * n_iter/(delta *n_nodes)))/(n_nodes* eps_tot)
	print("   Applying Bassily et al. bound, the sigma for Centralized DPSGD is ", sigma)
	return sigma


def net(L, n_nodes, eps_tot, delta, n_iter):
	# as usual
	K = ceil(conf * n_iter / n_nodes)
	delta_ = delta/K

	# bound on the spotted contributions
	K_spotted = ceil(K * 2/n_nodes + np.sqrt(6*K * np.log(1/delta)/n_nodes))
	print("ceci est Kspotted", K_spotted)

	# solve associate max eps_0
	def g(eps_0):
		eps_complex = np.sqrt(2*K_spotted*np.log(1/delta_))*eps_0 + K_spotted * eps_0 * (np.exp(eps_0)-1) 
		eps_simple = K_spotted * eps_0
		eps_true = min(eps_simple, eps_complex)
		return eps_true - eps_spotted

	beta = .5
	sigma_spotted = 50
	sigma_cand = 1

	while sigma_spotted/sigma_cand > 1.05 or sigma_cand/sigma_spotted > 1.05:
		
		eps_spotted = beta* eps_tot

		eps_0_spotted = optimize.bisect(g, 0, eps_spotted/np.sqrt(K_spotted))
		sigma_spotted = np.sqrt(2 *np.log(1.25/delta_))*L/eps_0_spotted
		eps_other = eps_tot - eps_spotted
		print("sigma due to spotted contribution", sigma_spotted)

		# case with numerical approximation
		if K < n_nodes * np.log(1/delta)/(2 * np.log(n_nodes)):
			print("   Non optimal regime for amplification, switching to numerical approx...")

			sigma_cand = np.sqrt(2 *np.log(1.25/delta))*L/eps_other

			def f(alpha):
				return L * np.sqrt(2 * alpha *(alpha - 1)) - sigma_cand

			while True:
				little =1e-5
				alpha = optimize.bisect(f, 1+little, (sigma_cand+10)/L)
				eps_assoc = 2 *K* L*L * alpha * np.log(n_nodes)/(sigma_cand**2 * n_nodes) + np.log(1/delta)/(alpha - 1)
				#print(sigma_cand, alpha, eps_assoc)
				if eps_assoc < eps_other:
					print("   The sigma for Network DPSGD is", sigma_cand)
					break
				else:
					sigma_cand*=1.15

		# case with exact formula
		else:
			sigma_cand = 4*L * np.sqrt(2 *K * np.log(n_nodes)* np.log(1/delta) / n_nodes ) / eps_other

		# adjusting the repartition between eps_cand and eps_other
		if sigma_spotted/sigma_cand > 1.05:
			print("here", sigma_spotted, sigma_cand)
			beta *= 1.01
		elif sigma_cand/sigma_spotted > 1.05:
			beta *= .99
			print("there", sigma_spotted, sigma_cand)
		else:
			print("beta", beta)
			return max (sigma_cand, sigma_spotted)

