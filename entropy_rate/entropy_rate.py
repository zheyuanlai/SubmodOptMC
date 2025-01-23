import numpy as np # type: ignore
from mc_generation import compute_stationary_distribution

def compute_entropy_rate(P):
    """
    Computes the entropy rate H(P) of the transition probability matrix P of a Markov chain.

    Parameters:
        P (ndarray): The transition probability matrix (2D numpy array).
        pi (ndarray): The marginal distribution (1D numpy array).

    Returns:
        float: The entropy rate H(P).
    """
    entropy_rate = 0.0
    pi = compute_stationary_distribution(P)
    for i, pi_x in enumerate(pi):
        for j, P_xy in enumerate(P[i]):
            if P_xy > 0:
                entropy_rate -= pi_x * P_xy * np.log(P_xy)
    return entropy_rate