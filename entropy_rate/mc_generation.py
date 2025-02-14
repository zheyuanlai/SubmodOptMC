import numpy as np # type: ignore
from itertools import product

def MC_generation(N, d, state_vals, eigenvalues, eigenfunction):
    '''
    Generates a reversible Markov chain with N eigenfunctions and d dimensions.

    Parameters:
        N (int): The number of eigenfunctions.
        d (int): The number of dimensions.
        state_vals (list): The values of the state space.
        eigenvalues (list): The eigenvalues.
        eigenfunction (function): The corresponding eigenfunction.

    Returns:
        tuple: The stationary distribution and the transition matrix.
    '''
    state_space = list(product(state_vals, repeat=d))
    state_space_size = len(state_space)
    #pi = np.ones(state_space_size) / state_space_size # set pi's to be uniform
    pi = np.linspace(0.01, 1, state_space_size); pi = pi / pi.sum() # set pi's to be non-uniform
    P = np.zeros((state_space_size, state_space_size))
    for i, x in enumerate(state_space):
        for j, y in enumerate(state_space):
            P[i, j] = sum(eigenvalues[n] * eigenfunction(n, x) * eigenfunction(n, y) * pi[j] for n in range(N))
    P = P / P.sum(axis=1, keepdims=True)
    return P

def compute_stationary_distribution(P):
    eigvals, eigvecs = np.linalg.eig(P.T)
    stationary_vector = np.real(eigvecs[:, np.isclose(eigvals, 1)])
    stationary_distribution = stationary_vector / stationary_vector.sum()
    return stationary_distribution.flatten()

def stationary_distribution_reversible(P):
    """
    Compute the stationary distribution of a reversible Markov chain.
    
    Parameters:
    P : numpy array (n x n)
        Transition probability matrix of the reversible Markov chain.
        
    Returns:
    pi : numpy array (n,)
        Stationary distribution.
    """
    n = P.shape[0]
    
    # Compute unnormalized stationary distribution using detailed balance
    pi_unnormalized = np.ones(n)  # Initialize with 1
    for i in range(1, n):
        pi_unnormalized[i] = pi_unnormalized[i-1] * (P[i-1, i] / P[i, i-1])
    
    # Normalize the stationary distribution
    pi = pi_unnormalized / np.sum(pi_unnormalized)
    
    return pi

### An example
def eigenfunction(n, x):
    #return np.prod([(1 + np.cos((n + 1) * np.pi * xi)) for xi in x])
    return np.prod([(1 + (n + 1) * np.pi * xi) for xi in x])

if __name__ == "__main__":
    N = 3
    d = 2
    state_vals = [0, 1]
    eigenvalues = [1, 0.5, 0.2]
    P = MC_generation(N, d, state_vals, eigenvalues, eigenfunction)
    print(P)