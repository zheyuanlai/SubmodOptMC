import numpy as np # type: ignore
from mc_generation import MC_generation
from keep_S_in import keep_S_in_mat
from leave_S_out import leave_S_out_mat
from outer_prod import compute_outer_product
from kl_div import KL_divergence
from submodular_minimizer import greedy

def submodular_function(S, P, state_vals):
    '''
    Compute the KL divergence for a given subset S.

    Parameters:
        S (set): The subset of dimensions to keep.
        P (ndarray): The original transition matrix.
        state_vals (list): The state values.

    Returns:
        float: The KL divergence D^\pi_{KL}(P || P^{(S)} ⊗ P^{(-S)}).
    '''
    P_S = keep_S_in_mat(P, state_vals, S)
    P_minus_S = leave_S_out_mat(P, state_vals, S)
    P_outer = compute_outer_product(P_S, P_minus_S)
    return KL_divergence(P, P_outer)

if __name__ == "__main__":
    N = 100
    d = 5
    state_vals = [0, 1]
    eigenvalues = [1 / (n + 1) for n in range(N)]
    eigenfunction = lambda n, x: np.prod([(1 + (n + 1) * np.pi * xi) for xi in x])

    P = MC_generation(N, d, state_vals, eigenvalues, eigenfunction)
    print(f"Generated multivariate reversible Markov chain with {d} dimensions.")

    submod_func = lambda S: submodular_function(S, P, state_vals)
    X = set(range(d))
    k = 2

    optimal_subset = greedy(submod_func, X, k)
    print(f"Optimal subset: {optimal_subset}")

    P_S = keep_S_in_mat(P, state_vals, optimal_subset)
    print(f"Optimal matrix: {P_S}")
    
    P_minus_S = leave_S_out_mat(P, state_vals, optimal_subset)
    P_outer = compute_outer_product(P_S, P_minus_S)
    opt_kl_div = KL_divergence(P, P_outer)
    print(f"Optimal KL divergence: {opt_kl_div}")

    non_optimal_subsets = [
        {i, i + 1} for i in range(d - 1)
        if {i, i + 1} != optimal_subset
    ]
    kl_divs = {tuple(S): KL_divergence(P, compute_outer_product(keep_S_in_mat(P, state_vals, S), leave_S_out_mat(P, state_vals, S))) for S in non_optimal_subsets}

    print(f"Entropy rates of non-optimal subsets: {kl_divs}")