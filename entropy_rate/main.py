from mc_generation import MC_generation, compute_stationary_distribution
from keep_S_in import keep_S_in_mat
from entropy_rate import compute_entropy_rate
from submodular_maximizer import greedy, lazy_greedy, stochastic_greedy
import numpy as np # type: ignore
from itertools import combinations
from visualization import simulate_path, plot_sample_paths

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

    submod_func = lambda S: compute_entropy_rate(keep_S_in_mat(P, state_vals, S))
    X = set(range(d))
    k = 2

    # Method 1: Greedy algorithm
    #optimal_subset = greedy(submod_func, X, k)
    #print("Greedy algorithm completed.")

    # Method 2: Lazy greedy algorithm
    optimal_subset = lazy_greedy(submod_func, X, k)
    print("Lazy greedy algorithm completed.")

    # Method 3: Stochastic greedy algorithm
    # Note: This algorithm may not provide the optimal result when `d` is small.
    #optimal_subset = stochastic_greedy(submod_func, X, k)
    #print("Stochastic greedy algorithm completed.")

    P_opt = keep_S_in_mat(P, state_vals, optimal_subset)
    opt_entropy_rate = compute_entropy_rate(P_opt)

    print(f"Optimal subset: {optimal_subset}")
    print(f"Optimal matrix: {P_opt}")
    print(f"Optimal entropy rate: {opt_entropy_rate}")

    # Visualization
    steps = 50
    original_path = simulate_path(P, state_vals, steps)
    subset_indices = sorted(optimal_subset)
    subset_path = simulate_path(P_opt, state_vals, steps, initial_state=original_path[0, subset_indices])
    plot_sample_paths(original_path, subset_path, list(optimal_subset))

    # Test the submodular optimization algorithm
    # Compare the entropy rates of the optimal subset and non-optimal subsets
    '''
    non_optimal_subsets = [
        set(combination) for combination in combinations(range(d), k)
        if set(combination) != optimal_subset
    ]
    entropy_rates = {tuple(S): compute_entropy_rate(keep_S_in_mat(P, state_vals, S)) for S in non_optimal_subsets}
    print(f"Entropy rates of non-optimal subsets: {entropy_rates}")
    '''