"""
experiment_unobserved_nodes.py

This script evaluates the performance of a neural ODE trained on a small graph
on larger graphs with a varying number of unobserved nodes.

Example:
We provide an example for evaluating a neural ODE trained on
timeseries data from the SIS model on a graph with moderate 
degree heterogeneity and clustering on larger graphs with the same 
properties.

M, Laber, 2026/02
"""

### IMPORT ###
import jax
import jax.numpy as jnp
import jax.experimental.sparse as sparse

import diffrax as dfx
import equinox as eqx
import numpy as np
import scipy
import os
import pickle

from graphs import *
from models import *
from trainers import *
from dynamics import *


## PARAMETERS ##

# i/o parameters
base_dir_graphs = f'./graphs/'                     # directory from which to load test graphs.
base_dir_checkpoints = f'./checkpoints/'           # directory from which to load the model.
base_dir_results = f'./results_unobserved_nodes/'  # directory at which to store results.

# misc parameters
seed_jax = 235  # seed of the random number generator. Needs to differ from training seed!
eps = 1e-15     # to avoid devision by zero when calculating the relative error

# parameters trained model
n_train = 64                    # number of nodes of the training graph
kbar_train = 10                 # average degree of the training graph      
gamma_train = 3.0               # degree exponent of the training graph   
beta_train = 1.1                # inverse temperature of the training graph 
graph_num_train = 0             # index of the training graph
noise_std_train = 0.000         # standard deviation of noise during training
layers = [1, 64, 64, 64, 1]     # layers of the neural ODE model
ngraphs_train = 10              # index of the first test graph

# graph parameters testing
n = 8192                         # number of nodes of the test graphs
kbar = kbar_train                # average degree of the test graphs
gamma = gamma_train              # degree exponent of the test graphs
beta =  beta_train               # inverse temperature of the test graphs
noise_std = 0.000                # standard deviation of noise during testing
ngraphs = 100                    # number of test graphs
ns_missing = np.logspace(        # different numbers of unobserved nodes during testing
    0,
    np.log2(n)-2,                # up to a quarter of nodes missing
    base=2,
    num=int(np.log2(n))-1,
    dtype=int
    )
               

# ode solver parameters 
solver_params = {'solver': dfx.Dopri5(),   # solver
                'atol': 1e-4,              # absolute tolerance
                'rtol': 1e-4,              # relative tolerance
                'dt0':  1e-5               # initial step size
                }

# dynamics parameters
dynamics = sis_model             # dynamical system to evaluate. Needs to be one of the dynamics defined in dynamics.py
pars_f = (1.0,)                  # parameters for the function f of the selected model. Set to (None,) if no parameters are needed.
pars_h_ego = (None,)             # parameters for the function h_ego of the selected model. Set to (None,) if no parameters are needed.
pars_h_alt = (1.2,)              # parameters for the function h_alt of the selected model. Set to (None,) if no parameters are needed.

t0 = 0                           # initial time        
t1 = 1.                          # final time
n_timesteps = 256                # number of time steps
n_samples = 1                    # number of initial conditions   
y0min = 0.                       # minimum value initial condition
y0max = 0.5                      # maximum value initial condition

## MAIN ##

# initialize the random number generator
key = jax.random.PRNGKey(seed_jax)
key, key_model, key_data, key_subset = jax.random.split(key, 4)

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

for n_missing in ns_missing:

    results = np.zeros(((ngraphs_train + ngraphs) * (n - n_missing), 17))
    
    for graph_num in range(ngraphs_train, ngraphs_train + ngraphs):
        
        print(f'n={n}, n_missing={n_missing}, graph={graph_num}')
        
        # load the graph
        graph_dir = f'{base_dir_graphs}n={n}_k={kbar}_gamma={gamma:.1f}_beta={beta:.1f}/graph_{graph_num:.0f}.pkl'

        with open(graph_dir, 'rb') as file:

            graph_data = pickle.load(file)

        edges = graph_data['edges']
        A = edges_to_adjacency(n, edges)

        # compute graph properties
        Asp = scipy.sparse.coo_matrix((A.data, (A.indices[:,0], A.indices[:, 1])), shape=A.shape)
        Asp = scipy.sparse.csr_matrix(Asp)

        # degree sequence
        k = np.array(Asp.sum(axis=0)).flatten()

        # clustering coefficient
        t = 0.5 * np.asarray((Asp @ Asp @ Asp).diagonal()).flatten()
        t_max = 0.5 * k * (k - 1)

        c = - np.ones_like(k)
        c[t_max > 0] = t[t_max > 0] / t_max[t_max > 0]
        c[t_max < 1] = np.nan

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
        
        if n_missing > 0:

            # select the observed subset of nodes
            key_subset, key_subset_ = jax.random.split(key_subset)
            observed = jax.random.choice(key_subset_, n, shape=(n - n_missing,), replace=False)

            observed = np.asarray(observed)
            rename_nodes = rename_nodes = {obs : i for i, obs in enumerate(observed)}

            select = np.isin(A.indices[:, 0], observed) & np.isin(A.indices[:, 1], observed)
            select_idx = np.asarray(A.indices[select])
            
            A_sub = sparse.BCOO((A.data[select], jnp.asarray([[rename_nodes[i], rename_nodes[j]] for i,j in select_idx])), shape=(n - n_missing, n - n_missing))
    
            # compute graph properties
            Asp = scipy.sparse.coo_matrix((A_sub.data, (A_sub.indices[:,0], A_sub.indices[:, 1])), shape=A_sub.shape)
            Asp = scipy.sparse.csr_matrix(Asp)
            
            # degree sequence
            k_sub = np.array(Asp.sum(axis=0)).flatten()
            
            # clustering coefficient
            t_sub = 0.5 * np.asarray((Asp @ Asp @ Asp).diagonal()).flatten()
            t_sub_max = 0.5 * k_sub * (k_sub - 1)
    
            c_sub = - np.ones_like(k_sub)
            c_sub[t_sub_max > 0] = t_sub[t_sub_max > 0] / t_sub_max[t_sub_max > 0]
            c_sub[t_sub_max < 1] = np.nan
            
            # make a prediction on the observed nodes
            ys_sub = ys_test[:, :, observed]
            ys_pred = jax.vmap(solve_nODE, in_axes=(None, 0, None))(ts, ys_sub[:, 0, :], ((None,), (None,), (None,), A_sub))
    
            # compute the node-wise loss
            mse_sub = jnp.mean(jnp.power(ys_sub - ys_pred, 2.), axis=1)
            mse_sub_mean = jnp.mean(mse_sub, axis=0)
            mse_sub_mom2 = jnp.mean(jnp.power(mse_sub, 2.), axis=0)
            
            # compute the node-wise absolute error
            mae_sub = jnp.mean(jnp.abs(ys_sub - ys_pred), axis=1)
            mae_sub_mean = jnp.mean(mae_sub, axis=0)
            mae_sub_mom2 = jnp.mean(jnp.power(mae_sub, 2.), axis=0)
    
            # compuate the node-wise relative error
            mre_sub = jnp.mean(jnp.abs(ys_sub - ys_pred) / (jnp.abs(ys_sub) + eps), axis=1)
            mre_sub_mean = jnp.mean(mre_sub, axis=0)
            mre_sub_mom2 = jnp.mean(jnp.power(mre_sub, 2.), axis=0)
            
        else:
                
            observed = np.arange(0, n)
            
            mse_sub_mean = mse_mean
            mae_sub_mean = mae_mean
            mre_sub_mean = mre_mean
            mse_sub_mom2 = mse_mom2
            mae_sub_mom2 = mae_mom2
            mre_sub_mom2 = mre_mom2
            

        # save node-wise data
        results[graph_num*(n-n_missing):(graph_num+1)*(n-n_missing), 0] = np.asarray(graph_num)
        results[graph_num*(n-n_missing):(graph_num+1)*(n-n_missing), 1] = np.asarray(k[observed])
        results[graph_num*(n-n_missing):(graph_num+1)*(n-n_missing), 2] = np.asarray(c[observed])
        results[graph_num*(n-n_missing):(graph_num+1)*(n-n_missing), 3] = np.asarray(mse_mean[observed])
        results[graph_num*(n-n_missing):(graph_num+1)*(n-n_missing), 4] = np.asarray(mae_mean[observed])
        results[graph_num*(n-n_missing):(graph_num+1)*(n-n_missing), 5] = np.asarray(mre_mean[observed])
        results[graph_num*(n-n_missing):(graph_num+1)*(n-n_missing), 6] = np.asarray(mse_mom2[observed])
        results[graph_num*(n-n_missing):(graph_num+1)*(n-n_missing), 7] = np.asarray(mae_mom2[observed])
        results[graph_num*(n-n_missing):(graph_num+1)*(n-n_missing), 8] = np.asarray(mre_mom2[observed])
        results[graph_num*(n-n_missing):(graph_num+1)*(n-n_missing), 9] = np.asarray(k_sub)
        results[graph_num*(n-n_missing):(graph_num+1)*(n-n_missing), 10] = np.asarray(c_sub)
        results[graph_num*(n-n_missing):(graph_num+1)*(n-n_missing), 11] = np.asarray(mse_sub_mean)
        results[graph_num*(n-n_missing):(graph_num+1)*(n-n_missing), 12] = np.asarray(mae_sub_mean)
        results[graph_num*(n-n_missing):(graph_num+1)*(n-n_missing), 13] = np.asarray(mre_sub_mean)
        results[graph_num*(n-n_missing):(graph_num+1)*(n-n_missing), 14] = np.asarray(mse_sub_mom2)
        results[graph_num*(n-n_missing):(graph_num+1)*(n-n_missing), 15] = np.asarray(mae_sub_mom2)
        results[graph_num*(n-n_missing):(graph_num+1)*(n-n_missing), 16] = np.asarray(mre_sub_mom2)

    # remove the initial zeros
    results = results[ngraphs_train*(n - n_missing):, :]

    # save the results
    results_dir = f'{base_dir_results}/model_n={n_train}_k={kbar_train}_gamma={gamma_train:.1f}_beta={beta_train:.1f}_graph={graph_num_train}_noise={noise_std_train:.3f}/'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)

    np.savez(
        f'{results_dir}result_n={n}_k={kbar}_gamma={gamma:.1f}_beta={beta:.1f}_noise={noise_std:.3f}_nmissing={n_missing}.npz',
            results
    )