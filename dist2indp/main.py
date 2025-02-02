from mc_generation import MC_generation, compute_stationary_distribution
#from keep_S_in import keep_S_in_mat
from leave_S_out import leave_S_out_mat
from entropy_rate import compute_entropy_rate
import numpy as np # type: ignore
from itertools import combinations
from supermodular_minimizer import greedy

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

    def supermod_func(S):
        res = 0
        for elem in S:
            res += compute_entropy_rate(leave_S_out_mat(P, state_vals, {elem}))
        res -= compute_entropy_rate(leave_S_out_mat(P, state_vals, S))
        return res
    
    X = set(range(d))
    k = 2

    optimal_subset = greedy(supermod_func, X, k)
    print("Greedy algorithm completed.")

    print(f"Optimal subset: {optimal_subset}")
    print(f"Optimal entropy rate: {supermod_func(optimal_subset)}")

    # Test the supermodular minimization algorithm
    non_optimal_subsets = [
        set(combination) for combination in combinations(range(d), k)
        if set(combination) != optimal_subset
    ]
    entropy_rates = {tuple(S): supermod_func(S) for S in non_optimal_subsets}
    print(f"Entropy rates of non-optimal subsets: {entropy_rates}")
