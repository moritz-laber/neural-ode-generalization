"""
dynamics.py

Defines the different dynamical systems in Barabasi-Barzel form.

M. Laber, 2026/02
"""


## IMPORT ##
import jax
import jax.numpy as jnp
from models import FactorizedNetworkDynamics

### SIS MODEL ###
# Defines the Susceptible-Infected-Susceptible model.
sis_model = FactorizedNetworkDynamics(
    f = lambda t, y, par : -par[0] * y,
    h_ego = lambda t, y, par : 1. - y,
    h_alt = lambda t, y, par : par[0] * y
)

### MAK MODEL ###
# Defines the Mass-Action-Kinetics model.
mak_model = FactorizedNetworkDynamics(
    f = lambda t, y, par : par[0] - par[1] * y,
    h_ego = lambda t, y, par : -y,
    h_alt = lambda t, y, par : par[0]*y
)

### MM MODEL ###
# Defines the Michaelis-Menten model.
mm_model = FactorizedNetworkDynamics(
    f = lambda t, y, par : -par[0] * jnp.power(y, par[1]),                                
    h_ego = lambda t, y, par : 1.,                                                       
    h_alt = lambda t, y, par : par[0] * jnp.power(y, par[1])/(1. + jnp.power(y,par[1]))
)

### BD MODEL ###
# Defines the Birth-Death-Process model.
bd_model = FactorizedNetworkDynamics(
    f = lambda t, y, par : - par[0] * jnp.power(y, par[1]),
    h_ego = lambda t, y, par : 1.,
    h_alt = lambda t, y, par : par[0] * jnp.power(y, par[1])
)

### ND MODEL ###
# Defines the Neuronal-Dynamics model.
nd_model = FactorizedNetworkDynamics(
    f = lambda t, y, par : - par[0] * y + par[1] * jnp.tanh(y),
    h_ego = lambda t, y, par : 1.,
    h_alt = lambda t, y, par : par[0]*jnp.tanh(y)
)

