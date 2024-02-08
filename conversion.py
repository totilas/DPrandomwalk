import numpy as np
from scipy import optimize

def renyi_to_eps_delta(delta, alpha, eps_alpha):
    """Return the eps of (eps, delta)-DP given the parameters alpha and eps_alpha of RDP"""
    return eps_alpha + np.log(1/delta)/(alpha-1)

def zcdp_to_eps_delta(z, delta):
    """Return the eps corresponding to the z"""
    return z + 2*np.sqrt(z*np.log(1/delta))

def eps_delta_to_zcdp(eps, delta):
    """Return the z corresponding to the pair eps delta"""
    def f(x):
        return zcdp_to_eps_delta(x, delta)-eps
    return optimize.bisect(f, 0, 2*eps)

