

from time import time
from typing import Callable, Iterable

import numpy as np
from matplotlib import pyplot as plt
import statsmodels.graphics.tsaplots as sm


#%%
from scipy import signal


def rwmh(ftilde: Callable, variances: np.ndarray, X0: np.ndarray,
         N: int, precond_const: float = 1.0, verbose: bool = True):
    """
    Random Walk Metropolis Hastings, i.e. type of Markov Chain Monte Carlo in continuous state space with Gaussian proposal
    density centered at current state.
    For a multidimensional state space, the components of the proposal samples are independent, i.e. the covariance
    matrix is diagonal.
    :param ftilde: un-normalized target density being the Markov Chain invariant distribution (after normalization)
    :param variances: 1D array of variances for each component of the proposal distribution
    :param X0: starting point of the chain
    :param N: chain length
    :param precond_const: constant for preconditionned Crank-Nicholson, e.g. sqrt(1 - s^2)
    :return: Markov Chain of length N
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
        Y += X[-1] * precond_const
        # Compute acceptance probability
        alpha = min(1., ftilde(Y) / ftilde(X[-1]))
        # Accept or reject
        if U < alpha:
            X.append(Y)
            accepted += 1
        else:
            X.append(X[-1].copy())

    p_accept = accepted / (N-1)
    if verbose:
        print(f'took {f"{time() - start:.3} s":<8} acceptance rate {p_accept:.3}')

    return np.array(X), p_accept


def diagnose_plot(chains: Iterable[np.ndarray], descriptions: Iterable[str], label: str, f=None):
    """
    Generate diagnosis plots for the given Markov Chain {X_n}, including:
        - histogram of {f(X_n)}
        - autocorrelation plot
        - traceplot

    :param chains: list of Markov chains
    :param descriptions: list (same size as chains) of textual descriptions for each chain
    :param label: x-label of histogram
    """
    _, axes = plt.subplots(len(chains), 3, figsize=(15, len(chains) * 3.5 + .5), sharex='col', sharey='col')
    for ax, chain, desc in zip(axes, chains, descriptions):
        ax[0].hist(chain, density=True, bins=40)
        sm.plot_acf(chain, ax=ax[1], lags=int(len(chain)/5))
        ax[2].plot(chain)
        ax[0].set_ylabel(desc)

    axes[0, 0].set_title('Histogram')
    axes[-1, 0].set_xlabel(label)
    axes[0, 1].set_title('Autocorrelation')
    axes[-1, 1].set_xlabel('Shift')
    axes[0, 2].set_title('Trace plot')
    axes[-1, 2].set_xlabel('Time step')
    plt.tight_layout()


def autocorrelation(x):
    c = signal.correlate(x, x)
    # scipy's correlate function also computes negative shifts, discard them
    midpoint = int((c.size - 1)/2)
    c = c[midpoint:]
    # Normalize
    c /= c[0]
    return c


def EES(chain, f=None):
    """Compute effective sample size"""
    if f is not None:
        raise NotImplementedError

    c = autocorrelation(chain)
    # Discard the autocorrelation at shift = 0
    c = c[1:]
    N = chain.shape[0]
    ees = N / (1 + 2 * c.sum())


if __name__ == '__main__':
    x = [np.random.randn(1000) for _ in range(5)]
    diagnose_plot(*x, label=r'$\theta_1$')

