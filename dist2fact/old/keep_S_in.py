import numpy as np  # type: ignore
from itertools import product
from mc_generation import stationary_distribution_reversible, MC_generation

def keep_S_in_mat(P, state_vals, S):
    '''
    Computes the keep-S-in matrix P_S on the subset of state space (S).

    Parameters:
        P (ndarray): The transition matrix of a reversible Markov chain.
        state_vals (list): The values of the state space.
        S (set): The subset of the state space.
    
    Returns:
        tuple: The marginal distribution pi_S and the keep-S-in matrix P_S.
    '''
    pi = stationary_distribution_reversible(P)
    d = int(np.log2(len(P)))
    state_space = list(product(state_vals, repeat=d))
    reduced_state_space = list(product(state_vals, repeat=len(S)))
    reduced_state_map = {state: idx for idx, state in enumerate(reduced_state_space)}

    pi_S = marginal_pi(S, P, state_space, reduced_state_space)
    P_S = np.zeros((len(pi_S), len(pi_S)))

    for i, x in enumerate(state_space):
        for j, y in enumerate(state_space):
            x_S = tuple(x[k] for k in S)
            y_S = tuple(y[k] for k in S)
            x_S_index = reduced_state_map[x_S]
            y_S_index = reduced_state_map[y_S]
            P_S[x_S_index, y_S_index] += pi[i] * P[i, j]

    for i, x_S in enumerate(pi_S):
        P_S[i, :] /= pi_S[i]
    return P_S

def marginal_pi(S, P, state_space, reduced_state_space):
    '''
    Computes the marginal distribution pi_S over reduced state space.

    Parameters:
        S (set): The subset of the state space.
        P (ndarray): The transition matrix of a reversible Markov chain.
        state_space (list): The state space.
        reduced_state_space (list): The reduced state space.
    
    Returns:
        ndarray: The marginal distribution pi_S over the reduced state space.
    '''
    pi = stationary_distribution_reversible(P)
    marg_dist = np.zeros(len(reduced_state_space))
    reduced_state_map = {state: idx for idx, state in enumerate(reduced_state_space)}

    for i, state in enumerate(state_space):
        partial_state = tuple(state[j] for j in S)
        partial_index = reduced_state_map[partial_state]
        marg_dist[partial_index] += pi[i]
    return marg_dist

if __name__ == "__main__":
    # example 1
    P = np.array([[0.333, 0.167, 0.333, 0.167],
                [0.167, 0.333, 0.167, 0.333],
                [0.333, 0.167, 0.333, 0.167],
                [0.167, 0.333, 0.167, 0.333]])
    S = {0}
    state_vals = [0, 1]
    print(keep_S_in_mat(P, state_vals, S))

    # example 2
    P = np.array([[0.4, 0.3, 0.2, 0.1],
                [0.1, 0.4, 0.3, 0.2],
                [0.2, 0.1, 0.4, 0.3],
                [0.3, 0.2, 0.1, 0.4]])
    S = {1}
    state_vals = [0, 1]
    print(keep_S_in_mat(P, state_vals, S))

    # example 3
    N = 2
    d = 5
    state_vals = [0, 1]
    eigenvalues = [1, 0.5]
    eigenfunction = lambda n, x: np.prod([(1 + np.cos((n + 1) * np.pi * xi)) for xi in x])
    P = MC_generation(N, d, state_vals, eigenvalues, eigenfunction)
    S = {0, 2, 4}
    print(keep_S_in_mat(P, state_vals, S))
