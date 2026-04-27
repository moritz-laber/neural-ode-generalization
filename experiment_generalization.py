"""
experiment_generalization.py

This script evaluates the neural ODE model trained on data from a specific 
dynamical system on a specific graph on test data from different graphs.

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
import numpy as np
import scipy
import pickle

from graphs import *
from models import * 
from trainers import *
from dynamics import *

## PARAMETERS ## 

# i/o parameters
base_dir_graphs = f'./graphs/'                   # directory from which to load test graphs.
base_dir_checkpoints = f'./checkpoints/'         # directory from which to load the model.
base_dir_results = f'./results_generalization/'  # directory at which to store results.

# misc parameters
seed_jax = 235  # seed of the random number generator. Needs to differ from the seed during training.
eps = 1e-15     # to avoid devision by zero when calculating the relative error

# parameters trained model
n_train = 64                    # number of nodes of the training graph
kbar_train = 10                 # average degree of the training graph      
gamma_train = 3.0               # degree exponent of the training graph   
beta_train = 1.1                # inverse temperature of the training graph 
graph_num_train = 0             # index of the training graph
noise_std_train = 0.000         # standard deviation of noise during training
layers = [1, 64, 64, 64, 1]     # layers of the neural ODE model
ngraphs_train = 10              # index of the first test graph.

# graph parameters testing
n = 8192                        # number of nodes of the test graphs    
kbar = 10                       # average degree of the test graphs
ngraphs = 100                   # number of test graphs for each parameter combination
noise_std = 0.000               # standard deviation of noise during testing

gammas = [2.1, 2.4, 2.7, 3.0, 3.3, 3.6, 3.9]            # degree exponents of the test graphs
betas = [0.1, 0.6, 1.1, 1.6, 2.1, 2.6, 3.1, 3.6, 4.1]   # inverse temperatures of the test graphs

# ode solver parameters 
solver_params = {
                'solver': dfx.Dopri5(),  # ODE solver
                'atol': 1e-4,            # absolute tolerance
                'rtol': 1e-4,            # relative tolerance
                'dt0':  1e-5             # initial step size
                }

# dynamics parameters
dynamics = sis_model            # dynamical system to evaluate. Needs to be one of the dynamics defined in dynamics.py
pars_f = (1.0,)                 # parameters for the function f of the selected model. Set to (None,) if no parameters are needed.
pars_h_ego = (None,)            # parameters for the function h_ego of the selected model. Set to (None,) if no parameters are needed.
pars_h_alt = (1.2,)             # parameters for the function h_alt of the selected model. Set to (None,) if no parameters are needed.
y0min = 0.                      # minimum value initial condition
y0max = 0.5                     # maximum value initial condition

t0 = 0                          # initial time        
t1 = 1.                         # final time
n_timesteps = 256               # number of time steps
n_samples = 1                   # number of initial conditions per graph  

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

for gamma in gammas:
    for beta in betas:
        results = []
        for graph_num in range(ngraphs_train, ngraphs_train + ngraphs):

            # load the graph
            graph_dir = f'{base_dir_graphs}n={n}_k={kbar}_gamma={gamma:.1f}_beta={beta:.1f}/graph_{graph_num:.0f}.pkl'
            
            with open(graph_dir, 'rb') as file:

                graph_data = pickle.load(file)

            edges = graph_data['edges']
            A = edges_to_adjacency(n, edges)

            # initialize the dynamics
            solve_dynamics = ODESolve(dynamics, solver_params)
            pars = (pars_f, pars_h_ego, pars_h_alt, A)

            # generate the test data
            ts = jnp.linspace(t0, t1, n_timesteps)
            key_data, key_test, key_noise  = jax.random.split(key_data, 3)
            y0_test, ys_test = generate_samples(
                key_test,
                n_samples,
                ts,
                (y0min, y0max),
                solve_dynamics,
                pars
            )

            # add noise
            if noise_std > 0:
                ys_test += noise_std*jax.random.normal(key_noise, shape=ys_test.shape)
            
            # make predictions on the test data
            ys_pred = jax.vmap(solve_nODE, in_axes=(None, 0, None))(ts, ys_test[:, 0, :], ((None,), (None,), (None,), A))

            # compute the node-wise loss
            mse = jnp.mean(jnp.power(ys_test - ys_pred, 2.), axis=1)
            mse_mean = jnp.mean(mse, axis=0)
            mse_mom2 = jnp.mean(jnp.power(mse, 2.), axis=0)
            
            # compute the node-wise absolute error
            mae = jnp.mean(jnp.abs(ys_test - ys_pred), axis=1)
            mae_mean = jnp.mean(mae, axis=0)
            mae_mom2 = jnp.mean(jnp.power(mae, 2.), axis=0)

            # compuate the node-wise relative error
            mre = jnp.mean(jnp.abs(ys_test - ys_pred) / (jnp.abs(ys_test) + eps), axis=1)
            mre_mean = jnp.mean(mre, axis=0)
            mre_mom2 = jnp.mean(jnp.power(mre, 2.), axis=0)

            # graph properties #
            A = scipy.sparse.coo_matrix((A.data, (A.indices[:,0], A.indices[:, 1])), shape=A.shape)
            A = scipy.sparse.csr_matrix(A)
            
            # degree sequence
            k = np.array(A.sum(axis=0)).flatten()
            
            # clustering coefficient
            t = 0.5 * np.asarray((A @ A @ A).diagonal()).flatten()
            t_max = 0.5 * k * (k - 1)

            c = - np.ones_like(k)
            c[t_max > 0] = t[t_max > 0] / t_max[t_max > 0]
            c[t_max < 1] = np.nan

            # save node-wise data
            results.append(
                jnp.stack([
                graph_num * jnp.ones(n),
                jnp.asarray(k),
                jnp.asarray(c),
                jnp.asarray(mse_mean),
                jnp.asarray(mae_mean),
                jnp.asarray(mre_mean),
                jnp.asarray(mse_mom2),
                jnp.asarray(mae_mom2),
                jnp.asarray(mre_mom2)
                ], axis=-1)
            )

        # remove the initial zeros
        results = jnp.concat(results, axis=0)

        # save the results
        results_dir = f'{base_dir_results}/model_n={n_train}_k={kbar_train}_gamma={gamma_train:.1f}_beta={beta_train:.1f}_graph={graph_num_train}_noise={noise_std_train:.3f}/'
        if not os.path.exists(results_dir):
            os.makedirs(results_dir, exist_ok=True)

        jnp.savez(
            f'{results_dir}result_n={n}_k={kbar}_gamma={gamma:.1f}_beta={beta:.1f}_noise={noise_std:.3f}.npz',
                results
        )
