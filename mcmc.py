

#%%
from time import time
from typing import Callable

import numpy as np


def rwmh(ftilde: Callable, variances: np.ndarray, X0: np.ndarray, N: int, verbose: bool = True):
    """
    Random Walk Metropolis Hastings, i.e. type of Markov Chain Monte Carlo in continuous state space with Gaussian proposal
    density centered at current state.
    For a multidimensional state space, the components of the proposal samples are independent, i.e. the covariance
    matrix is diagonal.
    :param ftilde: un-normalized target density being the Markov Chain invariant distribution (after normalization)
    :param variances: 1D array of variances for each component of the proposal distribution
    :param X0: starting point of the chain
    :param N: chain length
    :return:
    """
    start = time()
    X = [X0.copy()]
    # Generate uniform variables used for acceptance/rejection
    Us = np.random.random(N-1)
    # Generate the proposal samples (centered at zero) in advance, much more efficient -> shift them later
    Ys = np.random.multivariate_normal(np.zeros_like(X0), np.diag(variances), size=N-1)
    accepted = 0
    for U, Y in zip(Us, Ys):
        # Proposal sample properly shifted so that the Gaussian is centered around the current state
        Y += X[-1]
        # Compute acceptance probability
        alpha = min(1., ftilde(Y) / ftilde(X[-1]))
        # Accept or reject
        if U < alpha:
            X.append(Y)
            accepted += 1
        else:
            X.append(X[-1].copy())

    if verbose:
        p_accept = accepted / (N-1)
        print(f'took {f"{time() - start:.2} s":<8} acceptance rate {p_accept:.3}')

    return np.array(X)

