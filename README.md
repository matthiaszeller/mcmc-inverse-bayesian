
# MCMC Inverse Bayesian Problems in High Dimensions

## About

Repository of the exam project of the course [Stochastic Simulation MATH-414](https://edu.epfl.ch/studyplan/en/master/computational-science-and-engineering/coursebook/stochastic-simulation-MATH-414).

Given some noisy observation data being the output of a model (e.g. physical model), we want to find the parameters of 
the model. Under a *Bayesian framework*, we treat the model parameters as a random variable and we aim to compute the 
*posterior distribution*. The normalization constant of the posterior distribution being unknown, we will use *Markov 
Chain Monte Carlo* (MCMC) with Metropolis-Hastings algorithms to sample from the un-normalized density. 

This project investigates how different proposal distributions and how the dimension of the problem affect the 
sampling efficient.  


## Repository structure

* `mcmc.py` implements MCMC algorithms
* `utils.py` gathers routines and utility functions
* `project.ipynb` is the notebook running and analyzing simulations
* `report`: project report and slides
* `data`: stores simulation data


## Workflow

Asynchronous data generation and data analysis is achieved by using the utility functions `dump_simulation_results` and 
`load_simulation_results`, which writes and read simulation data in gzip-compressed json format. 
One can run several Markov Chains in parallel to leverage all CPU cores, as shown in the snippet below.

First define a function that will act a a worker of the multiprocessing pool, it simply runs the simulation and return 
the results with metadata:

```python
import numpy as np
from mcmc import pcn

def simulate(args):
    """Perform one simulation, used as map function for multiprocessor pool"""
    D, s = args
    X, p_accept = pcn(
        ftilde=lambda theta: ...,
        variances=...,
        X0=np.zeros(D),
        N=...,
        factor=np.sqrt(1 - s**2)
    )
    # Only return the first component of the chain, too much data otherwise
    return {
        'X': X[:, 0],
        'D': D,
        's': s,
        'p_accept': p_accept
    }
```

Then run several calls to the `simulate` function in paralell:
```python
import itertools
import multiprocessing as mp
import pandas as pd
from utils import dump_simulation_results, load_simulation_results

# Generate simluation data
D = (10, 100, 1000)
s = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
pool = mp.Pool(mp.cpu_count())
data = pool.map(simulate, itertools.product(D, s))
pool.close()
dump_simulation_results(data, 'data/file_name.json.gz')

# Load simulation data
df = pd.DataFrame(load_simulation_results('data/file_name.json.gz'))
```

