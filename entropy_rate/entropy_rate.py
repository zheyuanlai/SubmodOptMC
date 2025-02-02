import numpy as np # type: ignore
from mc_generation import compute_stationary_distribution

def compute_entropy_rate(P):
    """
    Computes the entropy rate H(P) of the transition probability matrix P of a Markov chain.
    Supports both 2D Markov chain entropy rate and 1D Shannon entropy.
    """
    if P.ndim == 1:
        return -np.sum(P * np.log(P + 1e-10))
    else:
        entropy_rate = 0.0
        pi = compute_stationary_distribution(P)
        for i, pi_x in enumerate(pi):
            for j, P_xy in enumerate(P[i]):
                if P_xy > 0:
                    entropy_rate -= pi_x * P_xy * np.log(P_xy)
        return entropy_rate