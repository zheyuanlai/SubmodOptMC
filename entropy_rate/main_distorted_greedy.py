from mc_generation import MC_generation, compute_stationary_distribution
from keep_S_in import keep_S_in_mat
from entropy_rate import compute_entropy_rate
import numpy as np # type: ignore
from itertools import combinations
from distorted_greedy import distorted_greedy

if __name__ == "__main__":
    N = 100
    d = 5
    state_vals = [0, 1]
    eigenvalues = [1 / (n + 1) for n in range(N)]
    def eigenfunction(n, x, alpha=0.3):
        damping = np.exp(-np.sum(alpha * np.array(x) ** 2))
        periodic_part = np.prod([1 + np.cos((n + 1) * np.pi * xi) for xi in x])
        return damping * periodic_part


    P = MC_generation(N, d, state_vals, eigenvalues, eigenfunction)
    print(f"Generated multivariate reversible Markov chain with {d} dimensions.")

    def submod_func(S):
        P_S = keep_S_in_mat(P, state_vals, S)
        pi_S = compute_stationary_distribution(P_S)
        return -compute_entropy_rate(pi_S * P_S)
    
    def modular_func(S):
        P_S = keep_S_in_mat(P, state_vals, S)
        pi_S = compute_stationary_distribution(P_S)
        return -compute_entropy_rate(pi_S)
    
    X = set(range(d))
    m = 2

    # Distorted Greedy Algorithm
    optimal_subset = distorted_greedy(submod_func, modular_func, X, m)
    print("Distorted greedy algorithm completed.")
    
    P_opt = keep_S_in_mat(P, state_vals, optimal_subset)
    opt_entropy_rate = compute_entropy_rate(P_opt)

    print(f"Optimal subset: {optimal_subset}")
    print(f"Optimal matrix: {P_opt}")
    print(f"Optimal entropy rate: {opt_entropy_rate}")

    non_optimal_subsets = [
        set(combination) for combination in combinations(range(d), m)
        if set(combination) != optimal_subset
    ]
    entropy_rates = {tuple(S): compute_entropy_rate(keep_S_in_mat(P, state_vals, S)) for S in non_optimal_subsets}
    print(f"Entropy rates of non-optimal subsets: {entropy_rates}")
