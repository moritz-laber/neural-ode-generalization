"""
experiment_stability.py

This script evaluates the fixed points and stability of the neural ODE model trained on data from a specific 
dynamical system and training graph, on other graphs.

Example:
We provide an example for evaluating a neural ODE trained on data from the SIS model
and a graph with moderate degree heterogeneity and clustering on data from graphs that
are larger an span a range of degree heterogeneities and clustering coefficients.

M. Laber, 2026/02
"""

### IMPORT ###

import os

import jax
import jax.numpy as jnp

import diffrax as dfx
import equinox as eqx
import pickle

from graphs import *
from models import *
from trainers import *
from dynamics import *

## PARAMETERS ##

# i/o parameters
base_dir_graphs = f'./graphs/'             # directory from which to load test graphs.
base_dir_checkpoints = f'./checkpoints/'   # directory from which to load the model.
base_dir_results = f'./results_stability/' # directory at which to store results. 

# misc parameters
seed_jax = 238                  # seed of the random number generator. Needs to be different from seed during training. 

# parameters trained model
n_train = 64                     # number of nodes of the training graph
kbar_train = 10                  # average degree of the training graph      
gamma_train = 3.0                # degree exponent of the training graph   
beta_train = 1.1                 # inverse temperature of the training graph 
graph_num_train = 0              # number of the training graph
layers = [1, 64, 64, 64, 1]      # layers of the neural ODE model
noise_std_train = 0.000          # standard deviation of the noise during training.
ngraphs_train = 10               # index of the first test graph.

# graph parameters testing
n = 4096                         # number of nodes of the test graphs
kbar = kbar_train                # average degree of the test graphs
gamma = gamma_train              # degree exponent of the test graphs
beta = beta_train                # inverse temperature of the test graphs
ngraphs = 100                    # number of test graphs for each parameter combination

# dynamics parameters
dynamics = sis_model             # dynamical system to evaluate. Needs to be one of the dynamics defined in dynamics.py
pars_f = (1.0,)                  # parameters for the function f of the selected model. Set to (None,) if no parameters are needed.
pars_h_ego = (None,)             # parameters for the function h_ego of the selected model. Set to (None,) if no parameters are needed.
pars_h_alt = (1.2,)              # parameters for the function h_alt of the selected model. Set to (None,) if no parameters are needed.
y0min = 0.0                      # lower limit for the initial conditions of the training samples.
y0max = 0.5                      # upper limit for the initial conditions of the training samples.

t0 = 0                           # initial time        
t1 = 10.                         # final time
n_timesteps = 256                # number of time steps
n_samples = 1                    # number of initial conditions per test graph

# ode solver parameters 
solver_params = {
                'solver': dfx.Dopri5(),  # ODE solver
                'atol': 1e-4,            # absolute tolerance
                'rtol': 1e-4,            # relative tolerance
                'dt0':  1e-5             # initial step size
                }

# Newton Method parameters
tol = 1e-4                       # tolerance for finding the fixed point with the Newton method
max_iter = 50                    # maximum number of iterations for finding the fixed point with the Newton method
    

## MAIN ##

# initialize the random number generator
key = jax.random.PRNGKey(seed_jax)
key, key_model, key_data = jax.random.split(key, 3)

# load the dynamics model
solve_dynamics = ODESolve(dynamics, solver_params)

# load the neural ODE model
key1, key2, key3 = jax.random.split(key_model, 3)

f_mlp = MLP(key1, layers, scale=1.)            # the scale for initialization does not matter because we are loading the weights
h_ego_mlp = MLP(key2, layers, scale=1.)
h_alt_mlp = MLP(key3, layers, scale=1.)

neural_ode = FactorizedNetworkDynamics(
    f=f_mlp,
    h_ego=h_ego_mlp,
    h_alt=h_alt_mlp
)

solve_nODE = ODESolve(neural_ode, solver_params)
solve_nODE = eqx.tree_deserialise_leaves(f'{base_dir_checkpoints}n={n_train}_k={kbar_train}_gamma={gamma_train:.1f}_beta={beta_train:.1f}_graph={graph_num_train}_noise={noise_std_train:.3f}/model_trained.eqx', solve_nODE)

        
results = []
problems = []  # stores graphs at which fixed point finding fails
for graph_num in range(ngraphs_train, ngraphs_train + ngraphs):

    print(f'n={n} k={kbar} gamma={gamma:.1f} beta={beta:.1f} graph {int(graph_num)}')

    # load the graph
    graph_dir = f'{base_dir_graphs}n={n}_k={kbar}_gamma={gamma:.1f}_beta={beta:.1f}/graph_{graph_num:.0f}.pkl'
    
    with open(graph_dir, 'rb') as file:
        graph_data = pickle.load(file)

    edges = graph_data['edges']
    A = edges_to_adjacency(n, edges)

    # create parameter set with new graph
    pars = (pars_f, pars_h_ego, pars_h_alt, A)

    # generate the test data
    ts = jnp.linspace(t0, t1, n_timesteps)
    key_data, key_test = jax.random.split(key_data, 2)
    y0_test, ys_test = generate_samples(
        key_test,
        n_samples,
        ts,
        (y0min, y0max),
        solve_dynamics,
        pars
    )

    # solve to get good guess for the fixed point
    try:
        ys_pred = jax.vmap(solve_nODE, in_axes=(None, 0, None))(ts, ys_test[:, 0, :], ((None,), (None,), (None,), A))
    except:
        problems.append(graph_num)
        continue

    # define helper functions to match function signature of solver
    def jac_dyn(y):

        return jacobian(y, solve_dynamics.vec, pars)

    def f_dyn(y):
    
        return solve_dynamics.vec(None, y, pars)
    
    def jac_nODE(y):
    
        return jacobian(y, solve_nODE.vec, pars)
    
    def f_nODE(y):
    
        return solve_nODE.vec(None, y, pars) 

    # compute the fixed points
    ystar_dyn, info_dyn = newton(f_dyn, jac_dyn, ys_test[0, -1, :], tol=tol, max_iter=max_iter)
    fystar_dyn = f_dyn(ystar_dyn)
    ystar_nODE, info_nODE = newton(f_nODE, jac_nODE, ys_pred[0, -1, :], tol=tol, max_iter=max_iter)
    fystar_nODE = f_nODE(ystar_nODE)

    # compute the Jacobian
    J_dyn = jacobian(ystar_dyn, solve_dynamics.vec, pars)
    J_nODE = jacobian(ystar_nODE, solve_nODE.vec, (None, None, None, A))

    # determine the spectrum of the Jacobian at the fixedpoint
    lam_dyn, _ = jnp.linalg.eig(jac_dyn(ystar_dyn).todense())
    lam_nODE, _ = jnp.linalg.eig(jac_nODE(ystar_nODE).todense())

    # save node-wise data
    results.append(
        jnp.stack(
            [graph_num * jnp.ones(n),
             ystar_dyn,
             fystar_dyn,
             ystar_nODE,
             fystar_nODE,
             jnp.real(lam_dyn),
             jnp.real(lam_nODE),
             jnp.imag(lam_dyn),
             jnp.imag(lam_nODE)],
            axis=-1
        )
    )

# concatenate the rows
results = jnp.concatenate(results, axis=0)

# save the results
results_dir = f'{base_dir_results}/model_n={n_train}_k={kbar_train}_gamma={gamma_train:.1f}_beta={beta_train:.1f}_graph={graph_num_train}_noise={noise_std_train:.3f}/'
if not os.path.exists(results_dir):
    os.makedirs(results_dir, exist_ok=True)

jnp.savez(
    f'{results_dir}result_n={n}_k={kbar}_gamma={gamma:.1f}_beta={beta:.1f}.npz',
    results
)

if len(problems)>0:
    jnp.savez(
        f'./problems_sis_n={n}_k={kbar}_gamma={gamma:.1f}_beta={beta:.1f}.npz',
        list(problems)
    )