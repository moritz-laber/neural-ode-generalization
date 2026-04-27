"""
trainers.py

This file gathers routines for training of neural ODEs on graphs.

M. Laber, 2026/02
"""

### IMPORT ###
import jax
import jax.numpy as jnp
from jax.random import PRNGKey
from jax.typing import ArrayLike
from jax.experimental import sparse

import diffrax as dfx
import equinox as eqx
import optax as tx

from typing import List, Tuple, Dict, Callable, Any
from tqdm import tqdm

### DATA LOADER FOR TIMESERIES DATA ###
def data_loader(key:PRNGKey, ys:ArrayLike, batch_size:int) -> ArrayLike:
    """Data loader for timeseries data. Loops infinitely over the data, shuffling along the batch dimension at each pass.
    
    Input
    key : PRNGKey for random number generation.
    ys : Training data in form of trajectories corresponding to the time points, shape (n_samples, n_timesteps, n_nodes).
    batch_size : Batch size for training.

    Output
    Generator yielding batches of training data with shape (batch_size, n_timesteps, n_nodes).
    """

    # extract the size of the data
    n_samples, n_timesteps, n_nodes = ys.shape

    # construct an index along the batch dimension
    batch_index = jnp.arange(n_samples)

    # allows for infinite looping over the data
    while True:

        # shuffle along the batch dimension
        key_, key = jax.random.split(key)
        permutation = jax.random.permutation(key_, n_samples)

        # initialize counters to know when one pass over the data
        # is over
        batch_start = 0
        batch_end = batch_size

        while batch_end < n_samples:

            # select the next batch of trajectories
            selected_batches = permutation[batch_start:batch_end]

            yield ys[selected_batches, :, :]

            # update counters
            batch_start += batch_size
            batch_end += batch_size

### TRAINING FOR NEURAL ODES ###
def train_neural_ode(key:PRNGKey, model:eqx.Module, ts:jnp.ndarray, ys:jnp.ndarray, args:Tuple[Tuple, Tuple, Tuple, sparse.BCOO], hyper_params:Dict[str, Any])-> Tuple[eqx.Module, List[float]]:
    """Train a neural ODE on timeseries data.
    
    Input
    key : PRNGKey for random number generation.
    model : Neural ODE model to be trained.
    ts : Time points at which the trajectories are evaluated.
    ys : Training data in form of trajectories corresponding to the time points, shape (n_samples, n_timesteps, n_nodes).
    args : Additional arguments to be passed to the model, (params_f, params_h_ego, params_h_alt, A), where A is the adjacency matrix.
    hyper_params : Dictionary containing hyperparameters for training, including:
        - optimizer: Optax optimizer to be used for training.
        - training_schedule: List of tuples (n_steps, trange) specifying the number of training steps and corresponding time range for each step.
        - batch_size: Batch size for training.
        - checkpoints: List of training steps at which to save model checkpoints.
        - checkpoint_path: Path to save model checkpoints.
    
    Output
    model : Trained neural ODE model.
    loss_list : List of loss values recorded during training.
    """

    @eqx.filter_value_and_grad
    def loss_val_grad(model, ts, ys, args):

        ypred = jax.vmap(model, in_axes=(None, 0, None))(ts, ys[:, 0, :], args)

        loss_val = jnp.power(jnp.mean(jnp.power(ys - ypred, 2.)), 1./2.)

        return loss_val
    
    @eqx.filter_jit
    def step(state, model, ts, ys, args):

        # calculate gradients and loss at the current step
        loss_val, grads = loss_val_grad(model, ts, ys, args)

        # calculate the parameter updates
        updates, state = optimizer.update(grads, state, model)

        # apply the updates to the model
        model = eqx.apply_updates(model, updates)

        return state, model, loss_val

    # training loop
    loss_list = []

    # initialize the optimizer
    optimizer = hyper_params['optimizer']
    state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    for n_steps, trange in hyper_params['training_schedule']:

        key, key_ = jax.random.split(key)
        loader = data_loader(key_, ys, batch_size=hyper_params['batch_size'])

        for i, ybatch in tqdm(zip(range(n_steps), loader)):

            # take a step in the optimization process and store the loss
            state, model, loss_val = step(state, model, ts[trange[0]: trange[1]], ybatch[:, trange[0]:trange[1], :], args)
            loss_list.append(loss_val.item())

            if i in hyper_params['checkpoints']:
                
                eqx.tree_serialise_leaves(f"{hyper_params['checkpoint_path']}model_step_{i}.eqx", model)

    return model, loss_list

### GENERATE SAMPLES ###
def generate_samples(key:PRNGKey, n_samples:int, ts:jnp.ndarray, y0bounds:Tuple[float, float], ode_solve:eqx.Module, pars:Tuple[Tuple, Tuple, Tuple, sparse.BCOO])-> Tuple[jnp.ndarray, jnp.ndarray]:
    """Generates samples of trajectories for a given dynamical system.
    
    Input
    key : PRNGKey for random number generation.
    n_samples : Number of initial conditions from which to generate trajectories.
    ts : Time points at which to evaluate the trajectories.
    y0bounds : Tuple specifying the lower and upper bounds for the initial conditions.
    ode_solve : A ODESolve instance mapping time points, initial conditions, and parameters to trajectories.
    pars : Tuple of parameter tuples and adjacency matrix (pars_f, pars_h_ego, pars_h_alt, A), where A is the adjacency matrix.

    Output
    y0 : Initial conditions from which trajectories were generated, shape (n_samples, n_nodes).
    ys : Generated trajectories corresponding to the initial conditions, shape (n_samples, n_timesteps, n_nodes).
    """

    # unpack initial conditions
    y0min, y0max = y0bounds

    # determine number of nodes
    n = pars[-1].shape[0]

    # sample initial conditions
    key, key_= jax.random.split(key)
    y0 = jax.random.uniform(key_, minval=y0min, maxval=y0max, shape=(n_samples, n))

    # solve ODE
    ys = jax.vmap(ode_solve, in_axes=(None, 0, None))(ts, y0, pars)

    return y0, ys




