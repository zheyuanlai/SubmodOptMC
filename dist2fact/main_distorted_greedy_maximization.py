from mc_generation import MC_generation, compute_stationary_distribution
from keep_S_in import keep_S_in_mat
from leave_S_out import leave_S_out_mat
import numpy as np # type: ignore
from itertools import combinations
from distorted_greedy import distorted_greedy
from kl_div import KL_divergence
from outer_prod import compute_outer_product
from visualization import simulate_path, plot_sample_paths

if __name__ == "__main__":
    N = 100
    d = 8
    state_vals = [0, 1]
    eigenvalues = [1 / (n + 1) for n in range(N)]
    def eigenfunction(n, x, alpha=0.3):
        damping = np.exp(-np.sum(alpha * np.array(x) ** 2))
        periodic_part = np.prod([1 + np.cos((n + 1) * np.pi * xi) for xi in x])
        return damping * periodic_part

    P = MC_generation(N, d, state_vals, eigenvalues, eigenfunction)
    print(f"Generated multivariate reversible Markov chain with {d} dimensions.")

    def c(S):
        res = 0
        for elem in S:
            res += KL_divergence(P, compute_outer_product(keep_S_in_mat(P, state_vals, {elem}), leave_S_out_mat(P, state_vals, {elem})))
        return -res

    def g(S):
        P_S = keep_S_in_mat(P, state_vals, S)
        P_minus_S = leave_S_out_mat(P, state_vals, S)
        return KL_divergence(P, compute_outer_product(P_S, P_minus_S)) + c(S)
        
    X = set(range(d))
    m = 3

    # Distorted Greedy Algorithm
    optimal_subset = distorted_greedy(g, c, X, m)
    print("Distorted greedy algorithm completed.")
    
    P_opt = keep_S_in_mat(P, state_vals, optimal_subset)
    dist2fact = KL_divergence(P, compute_outer_product(P_opt, leave_S_out_mat(P, state_vals, optimal_subset)))

    # Visualization
    steps = 50
    original_path = simulate_path(P, state_vals, steps)
    subset_indices = sorted(optimal_subset)
    subset_path = simulate_path(keep_S_in_mat(P, state_vals, optimal_subset), state_vals, steps, initial_state=original_path[0, subset_indices])
    plot_sample_paths(original_path, subset_path, list(optimal_subset))


    print(f"Optimal subset: {optimal_subset}")
    print(f"Optimal matrix: {P_opt}")
    print(f"Optimal entropy rate: {dist2fact}")

    non_optimal_subsets = [
        set(combination) for combination in combinations(range(d), m)
        if set(combination) != optimal_subset
    ]
    dist2facts = {tuple(S): KL_divergence(P, compute_outer_product(keep_S_in_mat(P, state_vals, S), leave_S_out_mat(P, state_vals, S))) for S in non_optimal_subsets}
    print(f"Entropy rates of non-optimal subsets: {dist2facts}")
