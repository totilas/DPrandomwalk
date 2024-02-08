import data
from sklearn.linear_model._base import LinearClassifierMixin, SparseCoefMixin, BaseEstimator
from sklearn.utils.extmath import log_logistic, safe_sparse_dot
from scipy.special import expit
from sklearn.utils.validation import check_X_y
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from numpy import linalg as LA
import random

def _intercept_dot(w, X, y):
    """taken from sklearn version 0.23.X (fd237278e895b42abe8d8d09105cbb82dc2cbba7)
    Computes y * np.dot(X, w).

    It takes into consideration if the intercept should be fit or not.

    Parameters
    ----------
    w : ndarray of shape (n_features,) or (n_features + 1,)
        Coefficient vector.

    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Training data.

    y : ndarray of shape (n_samples,)
        Array of labels.

    Returns
    -------
    w : ndarray of shape (n_features,)
        Coefficient vector without the intercept weight (w[-1]) if the
        intercept should be fit. Unchanged otherwise.

    c : float
        The intercept.

    yz : float
        y * np.dot(X, w).
    """
    c = 0.
    if w.size == X.shape[1] + 1:
        c = w[-1]
        w = w[:-1]

    z = safe_sparse_dot(X, w) + c
    yz = y * z
    return w, c, yz


def sgd(X, y, gamma, n_iter, obj_and_grad, theta_init, n_batch=1, freq_obj_eval=10,
        n_obj_eval=1000, random_state=None):
    """Stochastic Gradient Descent (SGD) algorithm

    Parameters
    ----------
    X : array, shape (n, d)
        The data
    y : array, shape (n,)
        Binary labels (-1, 1).
    gamma : float | callable
        The step size. Can be a constant float or a function
        that allows to have a variable step size
    n_iter : int
        The number of iterations
    obj_and_grad : callable
        A function which takes as a vector of shape (p,), a dataset of shape (n_batch, d)
        and a label vector of shape (n_batch,), and returns the objective value and gradient.
    theta_init : array, shape (p,)
        The initial value for the model parameters
    n_batch : int
        Size of the mini-batch to use at each iteration of SGD.
    freq_obj_eval : int
        Specifies the frequency (in number of iterations) at which we compute the objective
    n_obj_eval : int
        The number of points on which we evaluate the objective
    random_state : int
        Random seed to make the algorithm deterministic


    Returns
    -------
    theta : array, shape=(p,)
        The final value of the model parameters
    obj_list : list of length (n_iter / freq_obj_eval)
        A list containing the value of the objective function computed every freq_obj_eval iterations
    accuracy_list : list of length (n_iter / freq_obj_eval)
        A list containing the value of the test accuracy computed every freq_obj_eval iterations
    """
    
    rng = np.random.RandomState(random_state)
    n, d = X.shape
    p = theta_init.shape[0]
    
    theta = theta_init.copy()

    # if a constant step size was provided, we turn it into a constant function
    if not callable(gamma):
        def gamma_func(t):
            return gamma
    else:
        gamma_func = gamma
    
    # list to record the evolution of the objective (for plotting)
    obj_list = []
    # we draw a fixed subset of points to monitor the objective
    idx_eval = rng.randint(0, n, n_obj_eval)

    for t in range(n_iter):
        if t % freq_obj_eval == 0:
            # evaluate objective
            obj, _ = obj_and_grad(theta, X[idx_eval, :], y[idx_eval])
            obj_list.append(obj)
        
        idx_batch = rng.randint(0, n, n_batch)
        obj, grad = obj_and_grad(theta, X[idx_batch, :], y[idx_batch])
        theta -= gamma_func(t+1) * grad
    return theta, obj_list, accuracy_list




def my_logistic_obj_and_grad(theta, X, y, lamb):
    """Computes the value and gradient of the objective function of logistic regression defined as:
    min (1/n) \sum_i log_loss(theta;X[i,:],y[i]) + (lamb / 2) \|w\|^2,
    where theta = w (if no intercept), or theta = [w b] (if intercept)

    Parameters
    ----------
    theta_init : array, shape (d,) or (d+1,)
        The initial value for the model parameters. When an intercept is used, it corresponds to the last entry
    X : array, shape (n, d)
        The data
    y : array, shape (n,)
        Binary labels (-1, 1)
    lamb : float
        The L2 regularization parameter


    Returns
    -------
    obj : float
        The value of the objective function
    grad : array, shape (d,) or (d+1,)
        The gradient of the objective function
    """
    n_samples, n_features = X.shape
    grad = np.empty_like(theta)

    w, c, yz = _intercept_dot(theta, X, y)

    # Logistic loss is the negative of the log of the logistic function
    obj = -np.mean(log_logistic(yz)) + .5 * lamb * np.dot(w, w)

    z = expit(yz)
    z0 = (z - 1) * y

    grad[:n_features] = safe_sparse_dot(X.T, z0) / n_samples + lamb * w

    # Case where we fit the intercept
    if grad.shape[0] > n_features:
        grad[-1] = z0.sum() / n_samples
    return obj, grad


def chose_neigh(graph,u):
    return random.choices(np.arange(graph.shape[0]), weights=graph[u])[0]


def private_random_walk_sgd(X, y, gamma, n_iter, n_nodes, obj_and_grad, theta_init, graph,
    sigma=0,
    freq_obj_eval=10,
    n_obj_eval=1000,
    stopping_criteria = "contribute_then_noise",
    max_updates_per_node = 1,
    random_state=None,
    score=None,
    L=1,
    idx_node = 0,
):
    """Stochastic Gradient Descent (SGD) algorithm

    Parameters
    ----------
    X : array, shape (n, d)
        The data
    y : array, shape (n,)
        Binary labels (-1, 1).
    gamma : float | callable
        The step size. Can be a constant float or a function
        that allows to have a variable step size
    n_iter : int
        The number of iterations
    n_nodes : int
        number of nodes
    obj_and_grad : callable
        A function which takes as a vector of shape (p,), a dataset of shape (n_batch, d)
        and a label vector of shape (n_batch,), and returns the objective value and gradient.
    theta_init : array, shape (p,)
        The initial value for the model parameters
    graph: array, shape (n, n)
    sigma : float
        Standard deviation of the Gaussian noise added to each gradient
    freq_obj_eval : int
        Specifies the frequency (in number of iterations) at which we compute the objective
    n_obj_eval : int
        The number of points on which we evaluate the objective
    stopping_criteria : string
        If we carry on with noise or just stop
    max_updates_per_node : float
        Max number of updates per node authorized due to privay constraint
    random_state : int
        Random seed to make the algorithm deterministic
    score : callable 
        Score used to evaluate the model (in practice sklearn score on test set)
    L : float
        Max norm for the gradient (clipped to L)


    Returns
    -------
    theta : array, shape=(p,)
        The final value of the model parameters
    obj_list : list of length (n_iter / freq_obj_eval)
        A list containing the value of the objective function computed every freq_obj_eval iterations
    scores : list of length (n_iter / freq_obj_eval)
        A list containing the accuracy on the test set every freq_obj_eval iterations
    """

    if score is None:
        score = lambda c: 42
    
    rng = np.random.RandomState(random_state)
    n, d = X.shape
    p = theta_init.shape[0]
    
    theta = theta_init.copy()

    # if a constant step size was provided, we turn it into a constant function
    if not callable(gamma):
        def gamma_func(t):
            return gamma
    else:
        gamma_func = gamma
        
    
    n_updates = np.zeros(n_nodes)
    noisy = 0
        
    
    # list to record the evolution of the objective (for plotting)
    obj_list = []
    scores = []
    # we draw a fixed subset of points to monitor the objective
    idx_eval = rng.randint(0, n, n_obj_eval)
    sum_grad = 0
    samples_per_node = int(n/n_nodes)


    for t in range(n_iter):
        if t % freq_obj_eval == 0:
            # evaluate objective
            obj, _ = obj_and_grad(theta, X[idx_eval, :], y[idx_eval])
            obj_list.append(obj)

        # Draw a neighbor
        idx_node = chose_neigh(graph, idx_node)
        # Select all the samples belonging to this node (same size for all)
        idx = np.arange(idx_node*samples_per_node, (idx_node+1)*samples_per_node)
        # Compute the gradient on the private data of the node
        obj, grad = obj_and_grad(theta, X[idx, :], y[idx])
        # Noise to be added to the gradient
        shield = rng.normal(scale=sigma, size=p)
        # Stats to chose the clipping
        sum_grad += LA.norm(grad)

        # verifying the possible privacy constraint
        if stopping_criteria == "max_participation":
            # the algorithm stops as soon as one node reach the limit of participation
            n_updates[idx_node] += 1
            if n_updates[idx_node] > max_updates_per_node:
                print("iter", t, ": node", idx, "reached the maximum number of", max_updates_per_node, "updates")
                break
        
        elif stopping_criteria == "contribute_then_noise":
            # the node only adds noise if the maximum number of contributions has been reached (for Network DP)
            n_updates[idx_node] += 1
            if n_updates[idx_node] > max_updates_per_node:
                noisy += 1
                grad = 0
                # if noisy %200 == 0:
                    # print("iter", t, "there were ", noisy, "updates with only noise")
        
        elif stopping_criteria == "contribute_then_nothing":
            # the node only forwards the token if it already reached ist privacy budget (for Local DP)
            n_updates[idx_node] += 1
            if n_updates[idx_node] > max_updates_per_node:
               grad = 0
               shield = 0
               noisy+=1
               # if noisy %200 == 0:
                    # print("iter", t, "there were ", noisy, "updates with nothing")
               
        else:
            print("mistake in stopping criteria")
            break

        u = grad + shield
        # clipping the gradient
        if LA.norm(u) > L:
            u = L*u/LA.norm(u)
        # update model
        theta -= gamma_func(t+1) * (u)
        
        # computing score
        if t % freq_obj_eval == 0:
            scores.append(score(theta))


    # print("moyenne des normes des gradients", sum_grad/n_iter)
        
    return theta, obj_list, scores


class MyPrivateRWSGDLogisticRegression(BaseEstimator, LinearClassifierMixin, SparseCoefMixin):
    """Our sklearn estimator for private logistic regression defined as:
    min (1/n) \sum_i log_loss(theta;X[i,:],y[i]) + (lamb / 2) \|w\|^2,
    where theta = [w b]
    
    Parameters
    ----------
    gamma : float | callable
        The step size. Can be a constant float or a function
        that allows to have a variable step size
    n_iter : int
        The number of iterations
    nodes : int
        The number of nodes
    sigma : float
        Standard deviation of the Gaussian noise added to each gradient
    lamb : float
        The L2 regularization parameter    
    freq_obj_eval : int
        Specifies the frequency (in number of iterations) at which we compute the objective
    n_obj_eval : int
        The number of points on which we evaluate the objective
    stopping_criteria : string
        If we carry on with noise or just stop
    max_updates_per_node : float
        Max number of updates per node authorized due to privacy constraint
    random_state : int
        Random seed to make the algorithm deterministic
    score : callable 
        Score used to evaluate the model (in practice sklearn score on test set)
    L : float
        Max norm for the gradient (clipped to L)
        
    Attributes
    ----------
    coef_ : (p,)
        The weights of the logistic regression model.
    intercept_ : (1,)
        The intercept term of the logistic regression model.
    obj_list_: list of length (n_iter / freq_obj_eval)
        A list containing the value of the objective function computed every freq_loss_eval iterations
    """
    
    def __init__(self, gamma, n_iter, n_nodes,sigma,graph, lamb=0, freq_obj_eval=10, n_obj_eval=1000, stopping_criteria = "contribute_then_noise", max_updates_per_node = 1, random_state=None, score=lambda c: lambda d: 0, L=1):
        self.gamma = gamma
        self.n_iter = n_iter
        self.n_nodes = n_nodes
        self.sigma = sigma
        self.lamb = lamb
        self.freq_obj_eval = freq_obj_eval
        self.n_obj_eval = n_obj_eval
        self.stopping_criteria = stopping_criteria
        self.max_updates_per_node = max_updates_per_node
        self.random_state = random_state
        self.score = score
        self.L = L
        self.graph = graph
    
    def fit(self, X, y):
        
        # check data and convert classes to {-1,1} if needed
        X, y = self._validate_data(X, y, accept_sparse='csr', dtype=[np.float64, np.float32], order="C")
        self.classes_ = np.unique(y)    
        y[y==self.classes_[0]] = -1
        y[y==self.classes_[1]] = 1
                
        n, p = X.shape
        theta_init = np.zeros(p+1) # initialize parameters to zero
        # define the function for value and gradient needed by SGD
        obj_grad = lambda theta, X, y: my_logistic_obj_and_grad(theta, X, y, lamb=self.lamb)
        theta, obj_list, scores = private_random_walk_sgd(X, y,
            self.gamma,
            self.n_iter,
            self.n_nodes,
            obj_grad,
            theta_init,
            self.graph,
            self.sigma,
            self.freq_obj_eval,
            self.n_obj_eval,
            self.stopping_criteria,
            self.max_updates_per_node,
            self.random_state,
            score=self.score(np.unique(y)),
            L = self.L,
        )
        
        # save the learned model into the appropriate quantities used by sklearn
        self.intercept_ = np.expand_dims(theta[-1], axis=0)
        self.coef_ = np.expand_dims(theta[:-1], axis=0)
        
        # also save list of objective values during optimization for plotting
        self.obj_list_ = obj_list
        self.scores_ = scores
        
        return self
