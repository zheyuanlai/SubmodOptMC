import numpy as np # type: ignore
from mc_generation import stationary_distribution_reversible

def KL_divergence(P, Q):
    '''
    Compute the Kullback-Leibler divergence between two distributions P and Q.

    Parameters
    ----------
    P : numpy.ndarray
        The first matrix, which is the transition probability matrix of a reversible Markov chain.
    Q : numpy.ndarray
        The second matrix, which is the transition probability matrix of a reversible Markov chain.

    Returns
    -------
    float
        The Kullback-Leibler divergence between P and Q.
    '''
    kl_div = 0.0
    n = len(P)
    pi = stationary_distribution_reversible(P)
    
    for x in range(n):
        for y in range(n):
            if P[x, y] > 0 and Q[x, y] > 0:
                kl_div += pi[x] * P[x, y] * np.log(P[x, y] / Q[x, y])
    
    return kl_div
