"""
graphs.py

Provides routines for generating graphs from the S1 model, as well as utilities for working with these graphs in jax.

M. Laber, 2026/02
"""

### IMPORTS ###
import jax
import jax.numpy as jnp
import jax.experimental.sparse as sparse
import numpy as np
from numpy.typing import ArrayLike
import scipy
from typing import Dict, List, Tuple, Set

def pareto_inverse_cdf(u: ArrayLike, xbar: float, gamma: float) -> ArrayLike:
    """Inverse cumulative distribution function of the pareto distribution.
    
    Input
    u : values at which to evaluate the inverse cdf
    xbar : the mean of the Pareto distribution
    gamma : the exponent of the probability density function

    Output
    x : the values of the inverse cdf. If us is uniformly distributed on [0, 1],
        then x is Pareto distributed with mean xbar and exponent gamma.
    """

    return xbar * ((gamma - 2) / (gamma - 1)) * (1 - u)**(1 / (1 - gamma))

def _objective_function_cold(c: float, n: int, kbar: float, beta: float, chi_beta: ArrayLike) -> Tuple[float]:
    """Measures the deviation of the average degree from its expected value given a set of 
    coordinates and a chemical potential.
    
    Input
    c : the log chemical potential
    n : the number of nodes
    kbar : the expected degree
    beta : the inverse temperature
    chi_beta : log scaled hyperbolic distance exponentiated by inverse temperature
    
    Output
    r : residual of the optimization procedure
    dr : derivative of the residual of the optimization procedure   
    """

    # exponentiate for better readability
    c_beta = c**beta

    # calculate the connection probability
    p = 1. / (1. + chi_beta/c_beta)

    # calculate the derivative of the connection probability w.r.t. c
    dpdc = (beta/c) * (chi_beta/c_beta) / (1. + chi_beta/c_beta)**2

    # calculate the residual of the optimization
    r = 2. * np.sum(p) / n - kbar

    # calculate the change in residual w.r.t. c 
    drdc = 2.*np.sum(dpdc) / n

    return r, drdc

def _objective_function_hot(c: float, n: int, kbar: float, beta: float, theta: ArrayLike) -> Tuple[float]:
    """Measures the deviation of the average degree from its expected value given a set of 
    coordinates and a chemical potential.
    
    Input
    c : the log chemical potential
    n : the number of nodes
    kbar : the expected degree
    beta : the inverse temperature
    theta : the auxiliary variable theta
    
    Output
    r : residual of the optimization procedure
    dr : derivative of the residual of the optimization procedure   
    """

    # calculate the connection probability
    p = 1. / (1. + theta/c)

    # calculate the derivative of the connection probability w.r.t. c
    dpdc = (1./c)**2 * (theta / (1. + theta/c)**2)

    # calculate the residual of the optimization
    r = 2. * np.sum(p) / n - kbar

    # calculate the change in residual w.r.t. c 
    drdc = 2.*np.sum(dpdc) / n

    return r, drdc


def random_hyperbolic_graph(rng:np.random.Generator, n:int, kbar:float, gamma:float, beta:float, asymptotic:bool=False)->List[Tuple[int]]:
    """Random hyperbolic graph model.
    
    Input
    rng : random number generator
    n : number of nodes
    kbar  : average degree
    gamma : degree exponent
    beta  : inverse temperature

    Output
    edges : list edges
    """

    # sample coordinates uniformly on [0, 1]
    x = rng.uniform(low=0., high=1., size=n)

    # sample hidden degrees from the pareto distribution
    k = pareto_inverse_cdf(rng.uniform(size=n), kbar, gamma) 

    # cold regime
    if beta >= 1.0:
        # calculate chi
        chi_beta = np.zeros(n*(n-1)//2)
        for i in range(0, n-1):
            chi_beta[n*i - i*(i+1)//2 : n*i - i*(i+1)//2 +  n-i-1] = np.power(n*(1./2. - np.abs(1./2. - np.abs(x[i] - x[i+1:]))) / (k[i]*k[i+1:]), beta)
    
        # determine the log chemical potential (asymptotically)
        c0 = (np.sin(np.pi/beta)/(np.pi/beta)) / (2*kbar)

        if asymptotic:
            # use the asymptotic formula for the chemical potential
            c  = c0
        else:
            # determine the chemical potential exactly
            solution = scipy.optimize.root_scalar(_objective_function_cold, args=(n, kbar, beta, chi_beta), x0=c0, fprime=True, method='Newton')
            if solution.converged:
                c = solution.root
            else:
                print("The optimization procedure did not converge. Returning the initial guess.")
                c = c0
    
        # calculate the connection probability
        p = 1. / (1. + chi_beta / c**beta)
    
    # hot regime
    else:
        
        # determine the auxiliary variable theta
        theta = np.zeros(n*(n-1)//2)
        for i in range(0, n-1):
            theta[n*i - i*(i+1)//2 : n*i - i*(i+1)//2 +  n-i-1] = np.power(n*(1./2. - np.abs(1./2. - np.abs(x[i] - x[i+1:]))), beta)/(k[i]*k[i+1:])
        
        # determine the log chemical potential
        c0 = (1.0 - beta)/(np.power(2, beta) * kbar * np.power(n, 1.0 - beta ))

        if asymptotic:
            # use the asymptotic formula for the chemical potential
            c  = c0
        else:
            # determine the chemical potential exactly
            solution = scipy.optimize.root_scalar(_objective_function_hot, args=(n, kbar, beta, theta), x0=c0, fprime=True, method='Newton')
            if solution.converged:
                c = solution.root
            else:
                print("The optimization procedure did not converge. Returning the initial guess.")
                c = c0
        
        # calculate the connection probability
        p = 1. / (1. + theta / c)

    # sample the adjacency matrix
    a = rng.uniform(size=n*(n-1)//2) < p

    # convert to edge list
    edgelist = [(i, j) for i in range(0, n-1) for j in range(i+1, n) if a[n*i - i*(i+1)//2 + j - i - 1]>0]

    return edgelist

def edges_to_adjacency(n:int, edges:List[Tuple[int]], directed:bool=False, use_jax:bool=True):
    """Converts an edge list to an adjacency matrix.

    Input
    edges : list of edges as tuples.

    Output
    A : adjacency matrix.
    """

    # determine the number of edges
    m = len(edges)

    # convert to sparse adjacency matrix
    if use_jax:
        edges = jnp.asarray(edges)
        A = sparse.BCOO((jnp.ones(m), edges), shape=(n,n))
    else:
        edges = np.asarray(edges)
        A = scipy.sparse.coo_matrix((np.ones(m), (edges[:,0], edges[:,1])), shape=(n,n))
        A = scipy.sparse.csr_matrix(A)
    
    # make the network undirected
    if not directed:
        A = A + A.T

    return A