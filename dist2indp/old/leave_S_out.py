import numpy as np # type: ignore
from keep_S_in import keep_S_in_mat
from mc_generation import MC_generation

def leave_S_out_mat(P, state_vals, S):
    '''
    Computes the leave-S-out matrix P_{-S} by reusing the keep_S_in_mat function 
    and passing the complementary subset -S.

    Parameters:
        P (ndarray): The transition matrix.
        state_vals (list): The values of the state space.
        S (set): The subset of the state space to exclude.
    
    Returns:
        tuple: The marginal distribution pi_{-S} and the leave-S-out matrix P_{-S}.
    '''
    d = int(np.log2(len(P)))
    minus_S = {i for i in range(d) if i not in S}
    return keep_S_in_mat(P, state_vals, minus_S)

# Example usage
if __name__ == "__main__":
    # example 1
    P = np.array([[0.333, 0.167, 0.333, 0.167],
                  [0.167, 0.333, 0.167, 0.333],
                  [0.333, 0.167, 0.333, 0.167],
                  [0.167, 0.333, 0.167, 0.333]])
    S = {0}
    state_vals = [0, 1]
    print(leave_S_out_mat(P, state_vals, S))

    # example 2
    P = np.array([[0.4, 0.3, 0.2, 0.1],
                  [0.1, 0.4, 0.3, 0.2],
                  [0.2, 0.1, 0.4, 0.3],
                  [0.3, 0.2, 0.1, 0.4]])
    S = {1}
    state_vals = [0, 1]
    print(leave_S_out_mat(P, state_vals, S))

    # example 3
    N = 2
    d = 5
    state_vals = [0, 1]
    eigenvalues = [1, 0.5]
    eigenfunction = lambda n, x: np.prod([(1 + np.cos((n + 1) * np.pi * xi)) for xi in x])
    P = MC_generation(N, d, state_vals, eigenvalues, eigenfunction)
    S = {0, 2, 4}
    print(leave_S_out_mat(P, state_vals, S))
