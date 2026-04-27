"""
models.py

Gathers neural ODE model definitions and auxiliary functions for model evaluation.

M. Laber, 2026/02
"""

### IMPORTS ###
import jax
import jax.numpy as jnp
from jax.random import PRNGKey
from jax.typing import ArrayLike
from jax.experimental import sparse

import diffrax as dfx
import equinox as eqx
import optax as tx

from typing import List, Tuple, Dict, Callable, Any
from functools import partial

## LINEAR LAYER ##
class Linear(eqx.Module):

    w : ArrayLike
    b : ArrayLike

    def __init__(self, key:PRNGKey, shape:Tuple[int], scale:float=2.0) -> None:
        """Construct a linear layer that implements an affine transformation
           of the input.

        Input
        key   : key for the pseudo-random number generator
        shape : the shape of the weight matrix in the form (dim_in, dim_out)
        scale : scale factor for the initialization of the weights
        """

        # define input and output dimension
        dim_in, dim_out = shape

        # initialize the weights using Kaiming initialization
        self.w = jax.random.normal(key=key, shape=(dim_in, dim_out))*jnp.sqrt(scale/dim_in)

        # initialize the biases to zero
        self.b = jnp.zeros(shape=(dim_out,))

    def __call__(self, x:ArrayLike):
        """Compute the affine transformation of the input.

        Input
        x : input sample, shape (dim_in,)

        Output
        y : output sample after affine transformation, shape (dim_out,)
        """

        # affine transformation
        return (x @ self.w) + self.b

## MULTI-LAYER PERCEPTRON ##
class MLP(eqx.Module):

    layers : List
    activation : Callable

    def __init__(self, key:PRNGKey, dims:List, activation:Callable=jax.nn.selu, scale:int=2.0) -> None:
        """Construct a multilayer perceptron of arbitrary depth and width.

        Input
        key : key for the pseudo-random number generator
        dims : tuple of dimensions of the form (dim_in, dim_hidden_1, ..., dim_hidden_L, dim_out).
        activation : activation function to be applied elementwise after all but the last layer.
        """

        self.layers = []
        for i in range(len(dims) - 1):

            # initialize the linear transformations
            key, key_ = jax.random.split(key)
            self.layers.append(Linear(key_, shape=(dims[i], dims[i+1]), scale=scale))

        # set the activation function
        self.activation = activation

    def __call__(self, t, x:ArrayLike, args) -> ArrayLike:
        """Passes one sample through the MLP.

        Input
        x : input sample, shape (dim_in,)

        Output
        y : sample after passing through the MLP, shape (dim_out,)
        """

        x = jnp.asarray(x)

        for linear in self.layers[:-1]:
            
            # affine transformation
            x = linear(x)
            
            # non-linearity
            x = self.activation(x)

        # affine transformation in the last layer. No activation
        x = self.layers[-1](x)

        return x

### FactorizedNetworkDynamics ###
class FactorizedNetworkDynamics(eqx.Module):

    f : eqx.Module | Callable
    h_ego : eqx.Module | Callable
    h_alt : eqx.Module | Callable

    def __init__(self, f: eqx.Module | Callable, h_ego: eqx.Module | Callable, h_alt: eqx.Module | Callable) -> None:
        """Construct an instance of a dynamical system in Barabasi-Barzel form.

        Input
        f : self-dynamics
        h_ego : ego node contribution to the interaction term
        h_alt : alter node contribution to the interaction term             
        """

        self.f = f
        self.h_ego = h_ego
        self.h_alt = h_alt

    def __call__(self, t:float, y:jnp.ndarray, args:Tuple[Tuple, Tuple, Tuple, sparse.BCOO]) -> jnp.ndarray:
        """Evaluate the vector field of the dynamical system at a given time point and state.

        Input
        t : time point at which to evaluate the vector field.
        y : state at which to evaluate the vector field, shape (n_nodes,).
        args : additional arguments to be passed to the constituent functions of the vector field, (params_f, params_h_ego, params_h_alt, A), where A is the adjacency matrix.       
        
        Output
        dydt : vector field evaluated at the given time point and state, shape (n_nodes,).
        """

        pars_f, pars_h_ego, pars_h_alt, A = args

        if len(y.shape) == 1:
            y = y[None]

        fi = jax.vmap(self.f, in_axes=(None, 1, None))(t, y, pars_f).squeeze()
        hi = jax.vmap(self.h_ego, in_axes=(None, 1, None))(t, y, pars_h_ego).squeeze()
        hj = jax.vmap(self.h_alt, in_axes=(None, 1, None))(t, y, pars_h_alt).squeeze()

        return fi + hi * (hj @ A)


### NEURAL ODE ###
class ODESolve(eqx.Module):

    vec : eqx.Module | Callable
    solver : Callable
    rtol : float
    atol : float
    dt0 : float

    def __init__(self, vec:Callable, hyper_params:Dict) -> None:
        """Construct an ODESolve instance that can be used to solve a (neural) ODE.
        
        Input
        vec : vector field of the ODE to be solved, mapping time points, states, and additional arguments to the derivatives at the state.
        hyper_params : Dictionary containing hyperparameters for the ODE solver, including:
            - solver: Diffrax solver to be used for solving the ODE.
            - rtol: Relative tolerance for the ODE solver.
            - atol: Absolute tolerance for the ODE solver.
            - dt0: Initial step size for the ODE solver.
        """

        self.vec = vec
        self.solver = hyper_params['solver']
        self.rtol = hyper_params['rtol']
        self.atol = hyper_params['atol']
        self.dt0 = hyper_params['dt0']
        

    def __call__(self, ts, y0, args):
        """Solves the ODE defined by the vector field from a given initial condition and stores the solution at specified time points.
        
        Input
        ts : time points at which to store the solution, shape (n_timesteps,).
        y0 : initial condition for the ODE, shape (n_nodes,).
        args : additional arguments to be passed to the vector field, (params_f, params_h_ego, params_h_alt, A), where A is the adjacency matrix.

        Output
        ys : solution of the ODE at the specified time points, shape (n_timesteps, n_nodes).
        """

        terms = dfx.ODETerm(self.vec)
        saveat = dfx.SaveAt(ts=ts)
        stepsize_controller = dfx.PIDController(rtol=self.rtol, atol=self.atol)
        
        solution = dfx.diffeqsolve(
            terms=terms,
            solver=self.solver,
            t0=ts[0],
            t1=ts[-1],
            dt0=self.dt0, 
            y0=y0,
            args=args,
            saveat=saveat,
            stepsize_controller=stepsize_controller,
            max_steps=8192
        )
            
        return solution.ys

### JACOBIAN ###
def jacobian(ys:ArrayLike, vec:Callable| eqx.Module, pars:Tuple[Tuple, Tuple, Tuple, sparse.BCOO]) -> sparse.BCOO:
    """Computes the Jacobian of a dynamical system in Barabasi-Barzel form at a given state.
    
    Input
    ys : state at which to compute the Jacobian, shape (n_nodes,).
    vec : vector field of the dynamical system, mapping time points, states, and additional arguments to the derivatives at the state.
    pars : Tuple of parameter tuples and adjacency matrix (pars_f, pars_he, pars_ha, A), where A is the adjacency matrix.

    Output
    J : Jacobian of the vector field at the given state, shape (n_nodes, n_nodes) in sparse BCOO format.
    """

    # unpack parameters
    pars_f, pars_he, pars_ha, A = pars

    # compute the derivatives of the consituent functions of the vector field
    fprime = jax.jacfwd(lambda y: vec.f(None, y[None], pars_f))
    h_ego_prime = jax.jacfwd(lambda y: vec.h_ego(None, y[None], pars_he))
    h_alt_prime = jax.jacfwd(lambda y: vec.h_alt(None, y[None], pars_ha))

    
    # compute the off-diagonal elements of the Jacobian
    he = jax.vmap(vec.h_ego, in_axes=(None, -1, None))(None, ys[None], pars_he)
    ha_p = jax.vmap(h_alt_prime, in_axes=(-1))(ys)

    J = he * A * ha_p.T

    # compute the diagonal elements of the Jacobian
    f_p = jax.vmap(fprime, in_axes=(-1))(ys)
    he_p = jax.vmap(h_ego_prime, in_axes=(-1))(ys)
    ha = jax.vmap(vec.h_alt, in_axes=(None, -1, None))(None, ys[None], pars_ha)
    
    # fix dimensions if one of the constituent functions is a constant
    if f_p.ndim == 1:
        f_p = f_p[:, None]
    if he_p.ndim == 1:
        he_p = he_p[:, None]
    if ha.ndim == 1:
        ha = ha[:, None]

    diag_data = (f_p + he_p * (A @ ha)).flatten()
    diag = sparse.BCOO((diag_data, jnp.stack([jnp.arange(A.shape[0]), jnp.arange(A.shape[0])], axis=1)), shape=A.shape)

    # update the Jacobian by adding the diagonal elements
    J += diag

    return J

### FIXED POINT FINDING ###
@partial(jax.jit, static_argnums=(0,))
def armijo(f:Callable | eqx.Module, v: ArrayLike, y : ArrayLike, fy : ArrayLike, Jy : sparse.BCOO | ArrayLike, alpha0:float=1.0, c:float=1e-4, rho:float=0.5, max_iter:int=50)->float:
    """Subroutine checking the Armijo condition for a given stepsize and performing backtracking linesearch if the condition is not satisfied.
    
    Input
    f : function for which to find a root, mapping states to function values.
    v : search direction for the linesearch, shape (n_nodes,).
    y : current state, shape (n_nodes,).
    fy : function value at the current state, shape (n_nodes,).
    Jy : Jacobian of f at the current state, shape (n_nodes, n_nodes) in sparse BCOO format or dense array.
    alpha0 : initial stepsize for the Armijo linesearch.
    c : Armijo condition constant.
    rho : backtracking factor for the Armijo linesearch.
    max_iter : maximum number of iterations for the backtracking linesearch.
    
    Output
    alpha : stepsize satisfying the Armijo condition.
    """

    def continue_condition(state):

        alpha, i, satisfied = state

        return jnp.logical_and(i < max_iter, jnp.logical_not(satisfied))

    def step(state):

        alpha, i, _ = state
        ynew = y + alpha * v
        fynew = f(ynew)
        phi_new = 0.5 * jnp.sum(jnp.power(fynew, 2))
        phi_old = 0.5 * jnp.sum(jnp.power(fy, 2))

        satisfied = phi_new <= phi_old + c * alpha * jnp.dot(Jy.T @ fy, v)

        return alpha * rho, i + 1, satisfied

    # check initially
    _, _, satisfied0 = step((alpha0, 0, False))

    # backtracking linesearch
    alpha1, _, _ = jax.lax.while_loop(
        continue_condition,
        step,
        (alpha0 * rho, 0, satisfied0)
    )

    return jnp.where(satisfied0, alpha0, alpha1)

@partial(jax.jit, static_argnums=(0, 1))
def newton(f:Callable | eqx.Module, jac : Callable, y0 : ArrayLike, tol:float=1e-6, max_iter:int=100, alpha0:float=1.0, c:float=1e-4, rho:float=0.5):
    """Newton's method for finding a root of a function, with Armijo backtracking linesearch for determining the stepsize.
    
    Input
    f : function for which to find a root, mapping states to function values.
    jac : function mapping states to the Jacobian of f at the state, in sparse BCOO format.
    y0 : initial guess for the root, shape (n_nodes,).
    tol : tolerance for convergence, determined by the residual norm at the current state.
    max_iter : maximum number of iterations to perform.
    alpha0 : initial stepsize for the Armijo linesearch.
    c : Armijo condition constant.
    rho : backtracking factor for the Armijo linesearch.

    Output
    y1 : final iterate after convergence or reaching the maximum number of iterations, shape (n_nodes,).
    info : dictionary containing information about the optimization process, including:
        - iterations: number of iterations performed.
        - converged: boolean indicating whether convergence was achieved.
        - residual_norm: norm of the function value at the final iterate.
    """
    
    def newton_step(state):
    
        y, fy, i, _ = state
    
        # compute the Jacobian
        Jy = jac(y)
    
        # solve the linear system to get the Newton direction
        v = jnp.linalg.solve(Jy.todense(), -fy)
    
        # determine the stepsize with Armijo linesearch
        alpha = armijo(
            f,
            v,
            y,
            fy,
            Jy,
            alpha0=alpha0,
            c=c,
            rho=rho
        )
    
        # update state
        ynew = y + alpha * v
        fynew = f(ynew)
    
        # convergence check
        converged = jnp.linalg.norm(fynew) <= tol
    
        return ynew, fynew, i + 1, converged

    def continue_condition(state):
        y, fy, i, converged = state
        fy_norm = jnp.linalg.norm(fy)
        return jnp.logical_and(
            i < max_iter,
            jnp.logical_and(fy_norm > tol, jnp.logical_not(converged))
        )


    # Initial State
    fy0 = f(y0)
    initial_state = (y0, fy0, 0, False)   # (iterate, function value, iteration, converged)

    # Optimization Loop
    y1, fy1, iterations, converged = jax.lax.while_loop(
        continue_condition,
        newton_step,
        initial_state
    )

    info = {
        'iterations' : iterations,
        'converged' : converged,
        'residual_norm' : jnp.linalg.norm(fy1)
    }

    return y1, info