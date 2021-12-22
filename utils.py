import gzip
import json
from copy import deepcopy

import numpy as np


#%%

# NOT FASTER FOR BIG x, theta
# def u(x, theta):
#     """
#     Truncated sine-series expansion of log-permeability
#     :param x: position at which to evaluate log permeability
#     :param theta: Fourier coefficients
#     """
#     # Instead of doing a for loop, build 2D array for the terms pi*k*x, then matrix product with Fourier coeffs
#     # This is ~
#     k = np.arange(1, theta.size+1, dtype=float)
#     sine_argument = np.pi * np.outer(x, k)
#     res = (np.sqrt(2) / np.pi) * np.sin(sine_argument) @ theta
#     return res
from scipy import integrate


def u(x: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """
    Truncated sine-series expansion of log-permeability
    :param x: position at which to evaluate log permeability, 1D array
    :param theta: Fourier coefficients, 1D array
    """
    res = np.zeros_like(x)
    for k in range(1, theta.size+1):
        res += theta[k-1] * np.sin(np.pi * k * x)
    res *= np.sqrt(2) / np.pi
    return res


def G(theta: np.ndarray) -> np.ndarray:
    """
    Observation operator, returns the pressure (solution of PDE) evaluated at position x = [0.2, 0.4, 0.6, 0.8]
    The closed-form solution of the pressure is an integral, approximated by discretization with
    a number of intervals twice as big as the dimension of theta.
    :param theta: Fourier coefficients, 1D array
    :return: 1D array of length 4, pressure evaluated at positions [0.2, 0.4, 0.6, 0.8]
    """
    # Discretization step
    h = 1 / (2 * theta.size)
    # Grid at which to evaluate the integrand
    x = np.arange(0, 1+h, h)
    # Integrand array
    y = np.exp(-u(x, theta))
    # Integrate with trapezoidal rule
    cumtrpz = integrate.cumtrapz(y, dx=h, initial=0) # initial is completely useless, just simpler for indexing
    # Evaluate primitive at x = 0.2, 0.4, 0.6, 0.8. In order for those points to be exactly at a grid node,
    # the number of discretization intervals must be a multiple of 5.
    k, remainder = divmod(2*theta.size, 5) # n_interval = 5 * k + remainder
    if remainder != 0:
        raise ValueError('Number of subintervals must be a multiple of 5')
    # The values of interest are at index k, 2k, 3k, 4k
    p = 2 * cumtrpz[[k, 2*k, 3*k, 4*k]]
    # Normalize
    p /= cumtrpz[-1]
    return p


def target_density(theta: np.ndarray, sigma_noise: float, y_data: np.ndarray) -> float:
    """
    Un-normalized density of target distribution, that is the invariant distribution of the Markov Chain.
    :param theta: Fourier coefficients of log-permeability, passed to G function
    :param sigma_noise: noise standard deviation
    :param y_data: measured data
    :return: approximate density evaluated at theta for the given measured data
    """
    # Likelihood term
    log_likelihood = - 0.5 / sigma_noise**2 * ((y_data - G(theta)) ** 2).sum()
    # Prior term
    log_prior = - 0.5 * ( (theta * np.arange(1, theta.size+1))**2 ).sum()
    # Return posterior density
    return np.exp(log_likelihood + log_prior)


#%%

def dump_simulation_results(data, fpath):
    """Store simulation results, formatted as a list of dictionnaries.
    Each dictionnary has at least the entry `X`, an NxD array representing a Markov Chain,
    and additional fields that are useful for analysis"""
    # Make arrays json-serialization
    data = deepcopy(data)
    for dic in data:
        dic['X'] = dic['X'].tolist()
    # Dump in file
    with gzip.open(fpath, 'w') as f:
        json_data = json.dumps(data)
        f.write(json_data.encode('utf8'))


def load_simulation_results(fpath):
    """Load the data dumped by the `dump_simulation_results` function."""
    with gzip.open(fpath, 'r') as f:
        data = json.loads(f.read().decode())
    # Convert to numpy arrays
    for dic in data:
        dic['X'] = np.array(dic['X'])

    return data


#%%

if __name__ == '__main__':
    pass
    # # Debug u function
    # x = np.linspace(0, 1, 100)
    # theta = np.arange(1, 10, dtype=float) ** (-1)
    # res = u(x, theta)
    # res_debug = u_debug(x, theta)
    # assert (np.abs(res - res_debug) < 1e-15).all()

    theta = 1/np.arange(1, 11)
    G(theta)
