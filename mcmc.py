

from time import time
from typing import Callable, Iterable

import numpy as np
from matplotlib import pyplot as plt
import statsmodels.graphics.tsaplots as sm
from statsmodels.tsa.stattools import acf


#%%


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
    :return: Markov Chain of length N
    """
    start = time()
    X = [X0.copy()]
    # Generate uniform variables used for acceptance/rejection
    Us = np.random.random(N-1)
    # Generate the proposal samples (centered at zero) in advance, much more efficient -> shift them later
    Ys = np.random.multivariate_normal(np.zeros_like(X0), np.diag(variances), size=N-1)
    # Initialize loop
    accepted = 0
    ftilde_previous = ftilde(X[-1])
    for U, Y in zip(Us, Ys):
        # Proposal sample properly shifted so that the Gaussian is centered around the current state
        Y += X[-1]
        # Compute acceptance probability
        ftilde_proposal = ftilde(Y)
        alpha = min(1., ftilde_proposal / ftilde_previous)
        # Accept or reject
        if U < alpha:
            X.append(Y)
            accepted += 1
            ftilde_previous = ftilde_proposal
        else:
            X.append(X[-1].copy())
            # No need to update ftilde_previous

    p_accept = accepted / (N-1)
    if verbose:
        print(f'took {f"{time() - start:.3} s":<8} acceptance rate {p_accept:.3}')

    return np.array(X), p_accept



def pcn(ftilde: Callable, variances: np.ndarray, X0: np.ndarray, factor: float, N: int, verbose: bool = True):
    """
    Preconditionned Cranck-Nicolson.
    :param ftilde: un-normalized target density being the Markov Chain invariant distribution (after normalization)
    :param variances: 1D array of variances for each component of the proposal distribution
    :param X0: starting point of the chain
    :param N: chain length
    :return: Markov Chain of length N
    """
    start = time()
    X = [X0.copy()]
    # Generate uniform variables used for acceptance/rejection
    Us = np.random.random(N-1)
    # Generate the proposal samples (centered at zero) in advance, much more efficient -> shift them later
    Ys = np.random.multivariate_normal(np.zeros_like(X0), np.diag(variances), size=N-1)
    # Initialize loop
    accepted = 0
    ftilde_previous = ftilde(X[-1])
    for U, Y in zip(Us, Ys):
        # Proposal sample properly shifted so that the Gaussian is centered around the current state
        Y += X[-1] * factor
        # Compute acceptance probability
        ftilde_proposal = ftilde(Y)
        alpha = min(1., ftilde_proposal / ftilde_previous)
        # Accept or reject
        if U < alpha:
            X.append(Y)
            accepted += 1
            ftilde_previous = ftilde_proposal
        else:
            X.append(X[-1].copy())
            # No need to update ftilde_previous

    p_accept = accepted / (N-1)
    if verbose:
        print(f'took {f"{time() - start:.3} s":<8} acceptance rate {p_accept:.3}')

    return np.array(X), p_accept


def indep_sampler(ftilde: Callable, proposal_density: Callable, proposal_sampler: Callable,
                  X0: np.ndarray, N: int, verbose: bool = True):
    """
    MCMC independent sampling.
    :param ftilde: target distribution
    :param proposal_density: function taking 1 input and returning density of the proposal
    :param proposal_sampler: function taking 1 input (number of samples) and returns N iid samples from proposal
    :param X0: starting point
    :param N: chain length
    :param verbose: whether to print debugging informations
    :return:
    """
    start = time()
    X = [X0.copy()]
    # Generate uniform variables used for acceptance/rejection
    Us = np.random.random(N-1)
    # Generate the proposal samples
    Ys = proposal_sampler(N-1)
    # Initialize loop
    accepted = 0
    ftilde_previous = ftilde(X[-1])
    for U, Y in zip(Us, Ys):
        # Compute acceptance probability
        ftilde_proposal = ftilde(Y)
        alpha = min(1., ftilde_proposal / ftilde_previous * proposal_density(X[-1]) / proposal_density(Y))
        # Accept or reject
        if U < alpha:
            X.append(Y)
            accepted += 1
            ftilde_previous = ftilde_proposal
        else:
            X.append(X[-1].copy())
            # No need to update ftilde_previous

    p_accept = accepted / (N-1)
    if verbose:
        print(f'took {f"{time() - start:.3} s":<8} acceptance rate {p_accept:.3}')

    return np.array(X), p_accept


def diagnose_plot(chains: Iterable[np.ndarray], descriptions: Iterable[str], label: str, colors=None):
    """
    Generate diagnosis plots for the given Markov Chain {X_n}, including:
        - histogram of {f(X_n)}
        - autocorrelation plot
        - traceplot

    :param chains: list of Markov chains
    :param descriptions: list (same size as chains) of textual descriptions for each chain
    :param label: x-label of histogram
    """
    if colors is None:
        colors = [None] * len(chains)

    _, axes = plt.subplots(len(chains), 3, figsize=(15, len(chains) * 2.5 + .5), sharex='col', sharey='col')
    for ax, chain, desc, col in zip(axes, chains, descriptions, colors):
        ax[0].hist(chain, density=True, bins=30, color=col)
        ax[1].plot(acf(chain, nlags=chain.size-1, fft=True), color=col)
        ax[2].plot(chain, color=col)
        ax[0].set_ylabel(desc)

    axes[0, 0].set_title('Histogram')
    axes[-1, 0].set_xlabel(label)
    axes[0, 1].set_title('Autocorrelation')
    axes[-1, 1].set_xlabel('Shift')
    axes[0, 2].set_title('Trace plot')
    axes[-1, 2].set_xlabel('Time step')
    plt.tight_layout()


def ESS(chain: np.ndarray, debug: bool = False):
    """
    Compute effective sample size ESS = 1 / (1 + 2 * sum_{k>0} R[k]), with R[k] autocorrelation at lag k.
    Based on statsmodel's acf function to compute autocorrelation.
    The sum is truncated when autocorrelation becomes negative.

    :param chain: 1D array being (a component of) the chain
    :param debug: set True to also return autocorrelation and truncation point for debugging
    """
    c = acf(chain, nlags=chain.size - 1, fft=True)
    # Find first negative value
    truncation_index = np.argmax(c < 0.0) # argmax will stop at the first True value
    acf_sum = c[:truncation_index].sum()
    # Compute ESS
    ESS = chain.size / (1 + 2 * acf_sum)
    if np.isnan(c).any():
        ESS = 0.0

    if debug:
        return ESS, c, truncation_index
    return ESS


if __name__ == '__main__':
    #%%
    # Check ESS implementation
    xs = [
        np.random.randn(1000).cumsum() for _ in range(5)
    ]
    data = [ESS(x, debug=True) for x in xs]

    _, ax = plt.subplots(len(xs), 2, figsize=(15, 10))
    for a, (ess, c, trunc_idx) in zip(ax, data):
        a[0].plot(c)
        a[0].axvline(trunc_idx)
        print(ess)
        a[1].plot(np.cumsum(c))
        a[1].axvline(trunc_idx)

