#!/usr/bin/env python
"""
Optimized GPU-enabled version using vectorized operations.
The heavy linear-algebra computations are done in batch on the GPU,
minimizing Python-level loops.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from itertools import product, combinations

# ---- Device selection: prefer MPS if available, else CUDA, else CPU ----
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS device")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA device")
else:
    device = torch.device("cpu")
    print("Using CPU device")

# -----------------------
# Helper functions
# -----------------------

def get_state_space(state_vals, d):
    """
    Generates the full state space as a tensor of shape (num_states, d)
    using torch.cartesian_prod (vectorized).
    """
    if d == 0:
        return torch.empty((1, 0), dtype=torch.float32, device=device)
    grids = [torch.tensor(state_vals, dtype=torch.float32, device=device) for _ in range(d)]
    state_space = torch.cartesian_prod(*grids)  # shape: (|state_vals|^d, d)
    return state_space

def vectorized_eigenfunction(n, states, alpha=0.3):
    """
    Computes the eigenfunction for a given n in a vectorized way.
    `states` is a tensor of shape (num_states, d).
    """
    damping = torch.exp(-alpha * torch.sum(states**2, dim=1))
    periodic = torch.prod(1 + torch.cos((n + 1) * torch.pi * states), dim=1)
    return damping * periodic

# -----------------------
# Core functions
# -----------------------

def MC_generation(N, d, state_vals, eigenvalues):
    """
    Vectorized generation of the transition matrix P.
    Instead of 3 nested loops, we:
      1. Create a feature matrix F of size (num_states x N) where
         F[i, n] = eigenfunction(n, state_i).
      2. Compute P = (F @ (F * eigenvalues^T)) with an extra column weight.
    """
    states = get_state_space(state_vals, d)
    M = states.shape[0]
    
    pi = torch.linspace(0.01, 1.0, M, device=device)
    pi = pi / pi.sum()
    
    F = torch.empty((M, N), dtype=torch.float32, device=device)
    for n in range(N):
        F[:, n] = vectorized_eigenfunction(n, states)
    
    eigen_tensor = torch.tensor(eigenvalues, dtype=torch.float32, device=device)
    
    B = F * eigen_tensor
    temp = F @ B.t()
    P = temp * pi.unsqueeze(0)
    
    P = P / (P.sum(dim=1, keepdim=True) + 1e-10)
    return P

def compute_stationary_distribution(P):
    """
    Compute the stationary distribution of a reversible Markov chain
    using detailed balance (ratio method). This assumes the Markov chain
    has the structure (e.g. birth-death type) for which
       pi[i] = pi[i-1]*(P[i-1,i]/P[i,i-1])
    is valid.
    
    Parameters:
        P : torch.Tensor (n x n)
            Transition probability matrix of the reversible Markov chain.
        
    Returns:
        torch.Tensor (n,)
            Stationary distribution.
    """
    n = P.shape[0]
    pi_unnormalized = torch.ones(n, device=P.device, dtype=P.dtype)
    for i in range(1, n):
        pi_unnormalized[i] = pi_unnormalized[i-1] * (P[i-1, i] / P[i, i-1])
    pi = pi_unnormalized / torch.sum(pi_unnormalized)
    return pi

def keep_S_in_mat(P, state_vals, S):
    """
    Computes the reduced transition matrix P_S on the subset of dimensions S.
    This version uses vectorized operations for the full state space and constructs
    an explicit mapping from full states to reduced states using tuple conversion.
    """
    d = int(np.log2(P.shape[0]))
    full_states = get_state_space(state_vals, d)
    
    S_list = sorted(list(S))
    
    if len(S_list) == 0:
        num_reduced = 1
        full_to_reduced = torch.zeros(full_states.shape[0], dtype=torch.long, device=device)
    else:
        full_reduced = full_states[:, S_list]
        full_reduced_tuples = [tuple(int(round(x)) for x in row) for row in full_reduced.tolist()]
        reduced_tuples = list(product(state_vals, repeat=len(S_list)))
        mapping = {r: i for i, r in enumerate(reduced_tuples)}
        full_to_reduced = torch.tensor([mapping[r] for r in full_reduced_tuples],
                                        device=device, dtype=torch.long)
        num_reduced = len(reduced_tuples)
    
    pi_full = compute_stationary_distribution(P)
    M = torch.zeros((full_states.shape[0], num_reduced), dtype=torch.float32, device=device)
    M[torch.arange(full_states.shape[0]), full_to_reduced] = 1.0
    pi_S = M.t() @ pi_full  

    weighted_P = (pi_full.unsqueeze(1) * P)
    P_S_num = M.t() @ weighted_P @ M
    
    P_S = torch.zeros_like(P_S_num)
    for i in range(num_reduced):
        if pi_S[i] > 0:
            P_S[i, :] = P_S_num[i, :] / pi_S[i]
    return P_S

def leave_S_out_mat(P, state_vals, S):
    """
    Computes the leave-S-out matrix P_{-S} by reusing keep_S_in_mat 
    and passing the complementary subset -S.

    Parameters:
        P (torch.Tensor): The transition matrix on the GPU.
        state_vals (list): The values of the state space.
        S (set): The subset of the state space to exclude.
    
    Returns:
        torch.Tensor: The leave-S-out transition matrix P_{-S} on the GPU.
    """
    d = int(np.log2(P.shape[0]))
    minus_S = {i for i in range(d) if i not in S}
    return keep_S_in_mat(P, state_vals, minus_S)

def compute_entropy_rate(P):
    """
    Computes the entropy rate H(P) in a vectorized manner.
    """
    pi = compute_stationary_distribution(P)
    entropy_rate = -torch.sum(pi.unsqueeze(1) * P * torch.log(P + 1e-10))
    return entropy_rate

def compute_shannon_entropy(P):
    return -torch.sum(P * torch.log(P + 1e-10))

def compute_joint_entropy(joint):
    return -torch.sum(joint * torch.log(joint + 1e-10))

def distorted_greedy(f, c, U, m):
    """
    Implements the corrected Distorted Greedy algorithm.
    (Caching of f and c evaluations could yield further speedups.)
    """
    S = set()
    plot_res = []
    for i in range(m):
        max_gain = float('-inf')
        best_e = None
        for e in U - S:
            gain = (1 - 1/m) ** (m - (i + 1)) * (f(S | {e}) - f(S)) - c({e})
            if gain > max_gain:
                max_gain = gain
                best_e = e
        if best_e is not None and ((1 - 1/m) ** (m - (i + 1)) * (f(S | {best_e}) - f(S)) - c({best_e}) > 0):
            S.add(best_e)
        print(f"Subset size {i+1}: {S}")
        plot_res.append(compute_entropy_rate(keep_S_in_mat(P, state_vals, S)))
    return plot_res

def simulate_path(P, state_vals, steps, initial_state=None):
    """
    Simulates a path from a Markov chain defined by transition matrix P.
    (This part remains similar to your original code.)
    """
    d = int(np.log2(P.shape[0]))
    
    def state_to_index(state):
        return int("".join(map(str, state)), 2)
    
    def index_to_state(index):
        return list(map(int, f"{index:0{d}b}"))
    
    if initial_state is None:
        initial_state = np.random.choice(state_vals, size=d)
    
    current_state = initial_state
    states = [current_state]
    P_cpu = P.cpu().numpy()
    for _ in range(steps):
        current_index = state_to_index(current_state)
        transition_probs = P_cpu[current_index, :]
        next_index = np.random.choice(range(len(transition_probs)), p=transition_probs)
        current_state = index_to_state(next_index)
        states.append(current_state)
    return np.array(states)

def plot_sample_paths(original_path, subset_path, subset_indices):
    steps = original_path.shape[0]
    num_plots = len(subset_indices)
    fig, axes = plt.subplots(num_plots, 1, figsize=(10, 4 * num_plots), sharex=True)
    
    if num_plots == 1:
        axes = [axes]
    
    for i, idx in enumerate(subset_indices):
        axes[i].plot(range(steps), original_path[:, idx], label=f"Original MC (dim {idx})", linestyle="--")
        axes[i].plot(range(steps), subset_path[:, i], label=f"Subset MC (dim {i})", linestyle="-")
        axes[i].set_ylabel(f"State (dim {idx})")
        axes[i].legend()
    
    axes[-1].set_xlabel("Steps")
    plt.tight_layout()
    plt.savefig("sample_paths.png")
    plt.show()

def plot_objective_per_iteration(f_values):
    """
    Plots the function value f(S) after each iteration k = 0..m.
    f_values is a list of length (m+1).
    """
    iters = range(1, len(f_values) + 1)
    plt.figure(figsize=(6,4))
    plt.plot(iters, f_values, marker='o')
    plt.title("Optimal Entropy Rate vs. Subset Size")
    plt.xlabel("Subset Size")
    plt.ylabel("Optimal Entropy Rate")
    plt.grid(True)
    plt.show()

# -----------------------
# Main script
# -----------------------

if __name__ == "__main__":
    '''
    N = 100
    d = 5
    state_vals = [0, 1]
    eigenvalues = [1 / (n + 1) for n in range(N)]
    
    def eigenfunction(n, x, alpha=0.3):
        xt = torch.tensor(x, dtype=torch.float32, device=device)
        damping = torch.exp(-alpha * torch.sum(xt**2))
        periodic = torch.prod(1 + torch.cos((n + 1) * torch.pi * xt))
        return damping * periodic

    P = MC_generation(N, d, state_vals, eigenvalues)
    print(f"Generated multivariate reversible Markov chain with {d} dimensions.")
    
    # choose \beta = 0
    def modular_func(S):
        sum = 0
        for elem in S:
            sum += compute_entropy_rate(leave_S_out_mat(P, state_vals, {elem})) - compute_entropy_rate(P)
        return sum
    
    def submod_func(S):
        return compute_entropy_rate(keep_S_in_mat(P, state_vals, S)) + modular_func(S)
        
    X = set(range(d))
    m = 3
    
    optimal_subset = distorted_greedy(submod_func, modular_func, X, m)
    print("Distorted greedy algorithm completed.")
    
    P_opt = keep_S_in_mat(P, state_vals, optimal_subset)
    opt_entropy_rate = compute_entropy_rate(P_opt)
    
    print(f"Optimal subset: {optimal_subset}")
    print(f"Optimal matrix (P_opt):\n{P_opt.cpu().numpy()}")
    print(f"Optimal entropy rate: {opt_entropy_rate.item()}")

    #steps = 50
    #original_path = simulate_path(P, state_vals, steps)
    #subset_indices = sorted(optimal_subset)
    #subset_path = simulate_path(P_opt, state_vals, steps, initial_state=original_path[0, subset_indices])
    #plot_sample_paths(original_path, subset_path, list(optimal_subset))
    
    # Compute and print entropy rates for all non-optimal subsets of size m.
    non_optimal_subsets = [set(comb) for comb in combinations(range(d), m) if set(comb) != optimal_subset]
    entropy_rates = {}
    for S in non_optimal_subsets:
        entropy = compute_entropy_rate(keep_S_in_mat(P, state_vals, S))
        entropy_rates[tuple(sorted(S))] = entropy.item()
    print(f"Entropy rates of non-optimal subsets: {entropy_rates}")
    '''
    N = 100
    d = 5
    state_vals = [0, 1]
    eigenvalues = [1 / (n + 1) for n in range(N)]
    
    def eigenfunction(n, x, alpha=0.3):
        xt = torch.tensor(x, dtype=torch.float32, device=device)
        damping = torch.exp(-alpha * torch.sum(xt**2))
        periodic = torch.prod(1 + torch.cos((n + 1) * torch.pi * xt))
        return damping * periodic

    P = MC_generation(N, d, state_vals, eigenvalues)
    print(f"Generated multivariate reversible Markov chain with {d} dimensions.")
    
    # choose \beta = 0
    def modular_func(S):
        sum = 0
        for elem in S:
            sum += compute_entropy_rate(leave_S_out_mat(P, state_vals, {elem})) - compute_entropy_rate(P)
        return sum
    
    def submod_func(S):
        return compute_entropy_rate(keep_S_in_mat(P, state_vals, S)) + modular_func(S)
        
    X = set(range(d))
    function_values = []
    function_values = distorted_greedy(submod_func, modular_func, X, m=d)
    plot_objective_per_iteration([f.item() for f in function_values])