"""
generate_graphs.py

Generate graphs from the S1 model and store them.

Example:
We provide an example for generating graphs with
8192 nodes, average degree 10, in a range of 
values of degree exponent and inverse temperature.

M. Laber, 2026/02
"""

## IMPORT ##
from datetime import datetime
from graphs import random_hyperbolic_graph
import numpy as np
import os
import pickle

## PARAMETERS ##
base_dir = './graphs/'  # directory at which to store the generated graphs

n = 8192                                     # number of nodes
GAMMA = [2.1, 2.4, 2.7, 3.0, 3.3, 3.6, 3.9]  # values of the degree exponent 
BETA =  [1.1, 1.6, 2.1, 2.6, 3.1, 3.6, 4.1]  # values of the inverse temperature
NGRAPHS = 110                                # number of graphs to generate for each parameter combination
KBAR = 10                                    # average degree


## MAIN ##
for gamma in GAMMA:
    for beta in BETA:
        
        if not os.path.exists(base_dir+f"n={n}_k={KBAR}_gamma={gamma}_beta={beta}"):
            os.mkdir(base_dir+f"n={n}_k={KBAR}_gamma={gamma}_beta={beta}")
        
        for i in range(NGRAPHS):

            # draw the random number generator seed
            now = datetime.now()
            seed = int((now.hour * 3600 + now.minute * 60 + now.second) * 1000 + now.microsecond // 1000)

            # generate a random hyperbolic graph
            edgelist = random_hyperbolic_graph(np.random.default_rng(seed), n, KBAR, gamma , beta, asymptotic=False)

            # store results
            data_dict = {
                'n'     : n,
                'kbar'  : KBAR,
                'gamma' : gamma,
                'beta'  : beta,
                'edges' : edgelist,
                'rng'   : seed
            }

            with open(base_dir+f"n={n}_k={KBAR}_gamma={gamma}_beta={beta}/graph_{i}.pkl", "wb") as file:

                pickle.dump(data_dict, file)


            

