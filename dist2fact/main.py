import numpy as np # type: ignore
from mc_generation import MC_generation
from keep_S_in import keep_S_in_mat
from leave_S_out import leave_S_out_mat
from outer_prod import compute_outer_product
from kl_div import KL_divergence
from submodular_minimizer import greedy
from itertools import combinations
from visualization import simulate_path, plot_sample_paths

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
    #eigenfunction = lambda n, x: np.prod([(1 + np.cos((n + 1) * np.pi * xi)) for xi in x])
    #eigenfunction = lambda n, x: np.prod([(1 + (n + 1) * np.pi * xi) for xi in x])
    def eigenfunction(n, x, alpha=0.3):
        damping = np.exp(-np.sum(alpha * np.array(x) ** 2))
        periodic_part = np.prod([1 + np.cos((n + 1) * np.pi * xi) for xi in x])
        return damping * periodic_part

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

    # Visualization
    steps = 50
    original_path = simulate_path(P, state_vals, steps)
    subset_indices = sorted(optimal_subset)
    subset_path = simulate_path(P_S, state_vals, steps, initial_state=original_path[0, subset_indices])
    plot_sample_paths(original_path, subset_path, list(optimal_subset))

    # Test the submodular minimization algorithm
    # Compare the KL divergences of the optimal subset and non-optimal subsets
    '''
    non_optimal_subsets = [
        set(combination) for combination in combinations(range(d), k)
        if set(combination) != optimal_subset
    ]
    kl_divs = {
        tuple(S): submod_func(S)
        for S in non_optimal_subsets
    }

    print(f"Entropy rates of non-optimal subsets: {kl_divs}")
    '''