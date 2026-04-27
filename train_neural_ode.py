"""
train_neural_ode.py

Trains a neural ODE for a given dynamical systems from dynamics.py.

Example:
We provide an example for training a neural ODE on
timeseries data from the SIS model on a graph with 
moderate degree heterogeneity and clustering.


M. Laber, 2026/02
"""

## IMPORT ##
import jax
import jax.numpy as jnp

import optax as otx
import diffrax as dfx
import equinox as eqx
import numpy as np
import os
import sys
import pickle
import warnings

from graphs import *
from models import *
from trainers import *
from dynamics import *

## PARAMETERS ##

# i/o parameters
base_dir_graphs = f'./graphs/'               # directory from which to load training graphs.
base_dir_checkpoints = f'./checkpoints/'     # directory at which to store checkpoints.

# Graph Parameters
n = 64             # number of nodes of the training graph.
kbar = 10          # average degree of the training graph.
gamma = 3.0        # degree exponent of the training graph.
beta = 1.1         # inverse temperature of the training graph.
graph_num = 0      # index of the graph to load for training.

# Dataset Parameters
t0 = 0                            # lower limit for the time interval of the training samples.
t1 = 1.                           # upper limit for the time interval of the training samples.
n_timesteps = 256                 # number of time points at which to evaluate the dynamics for the training samples.
n_samples = 256                   # number of training samples to generate.
noise_std = 0.000                 # the standard deviation of independent, additive Gaussian noise with zero mean. Set to 0 for noiseless training.

dynamics = sis_model         # set to one of the dynamics defined in dynamics.py
pars_f = (1.0,)              # parameters for the function f of the selected model. Set to (None,) if no parameters are needed.
pars_h_ego = (None,)         # parameters for the function h_ego of the selected model. Set to (None,) if no parameters are needed.
pars_h_alt = (1.2,)          # parameters for the function h_alt of the selected model. Set to (None,) if no parameters are needed.

y0min = 0.                   # lower limit for the initial conditions of the training samples.
y0max = 0.5                  # upper limit for the initial conditions of the training samples.

# Neural ODE & Training Parameters
scale = 1e-2           # scale for the initialization of MLP weights.
seed_jax = 121         # seed for weight initialization, training data, and training procedure.

learning_rate = 1e-3   # learning rate scaling factor for the optimizer, actual learning rate is adaptive.
batch_size = 32        # batch size for training.

layers = [1, 64, 64, 64, 1]                 # number of neurons per layer for the 3 MLPs.
solver_params = {
                 'solver': dfx.Dopri5(),    # ODE solver
                 'atol': 1e-4,              # absolute tolerance
                 'rtol': 1e-4,              # relative tolerance
                 'dt0': 1e-5                # initial step size
                }

n_optimizersteps = 3e5                                         # number of optimization steps for training.
n_checkpoints = 100                                            # number of checkpoints to save during training.
trainer_params = {                                             
                  'optimizer': otx.adagrad(learning_rate),     # optimizer for training.
                  'batch_size':batch_size,
                  'training_schedule': [(int(n_optimizersteps), (0, n_timesteps))],
                  'checkpoints' : [int(i) for i in np.logspace(1, np.log10(n_optimizersteps), n_checkpoints, base=10)],
                  'checkpoint_path' : ''
                 }

### MAIN ###

## test gpu availability ##
gpu_devices = [device for device in jax.devices() if device.platform=='gpu']
if not gpu_devices:
    warnings.warn("No GPU available.")

## Initialize random number generators 
key = jax.random.PRNGKey(seed_jax)
key, key_model, key_data, key_training = jax.random.split(key, 4)

## Load the Graph
graph_dir = f'{base_dir_graphs}n={n}_k={kbar}_gamma={gamma:.1f}_beta={beta:.1f}/graph_{graph_num}.pkl'
with open(graph_dir, 'rb') as file:

    graph_data = pickle.load(file)

edges = graph_data['edges']
A = edges_to_adjacency(n, edges)

## Generate Training Data
ts = jnp.linspace(t0, t1, n_timesteps)

solve_dynamics = ODESolve(dynamics, solver_params)
pars = (pars_f, pars_h_ego, pars_h_alt, A)

key_train, key_noise  = jax.random.split(key_data)
y0_train, ys_train = generate_samples(
    key_train,
    n_samples,
    ts,
    (y0min, y0max),
    solve_dynamics,
    pars
)

# add noise (optional)
if noise_std > 0:
    ys_train += noise_std*jax.random.normal(key_noise, shape=ys_train.shape)

# update the checkpoint directory
checkpointpath = f'{base_dir_checkpoints}n={n}_k={kbar}_gamma={gamma:.1f}_beta={beta:.1f}_graph={graph_num}_noise={noise_std:.3f}/'
if not os.path.exists(checkpointpath):
        os.makedirs(checkpointpath, exist_ok=True)

trainer_params['checkpoint_path'] = checkpointpath

## Initialize the model and save at initialization
key1, key2, key3 = jax.random.split(key_model, 3)

f_mlp = MLP(key1, layers, scale=scale)
h_ego_mlp = MLP(key2, layers, scale=scale)
h_alt_mlp = MLP(key3, layers, scale=scale)

neural_ode = FactorizedNetworkDynamics(
    f =f_mlp,
    h_ego =  h_ego_mlp,
    h_alt =  h_alt_mlp
)

solve_nODE = ODESolve(neural_ode, solver_params)

eqx.tree_serialise_leaves(trainer_params['checkpoint_path'] + 'model_untrained.eqx', solve_nODE)

# Train the neural ODE and save at checkpoints, and fully trained
solve_nODE, loss_list = train_neural_ode(
    key_training,
    solve_nODE,
    ts,
    ys_train,
    ((None,), (None,), (None,), A),
    trainer_params
)

eqx.tree_serialise_leaves(trainer_params['checkpoint_path'] + 'model_trained.eqx', solve_nODE)

with open(trainer_params['checkpoint_path'] + 'training_loss.pkl', 'wb') as file:

    pickle.dump(loss_list, file)
