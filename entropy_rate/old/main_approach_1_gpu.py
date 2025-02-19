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
    # When d is 0, return a tensor with shape (1,0)
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
    # Create full state space (num_states x d)
    states = get_state_space(state_vals, d)
    M = states.shape[0]
    
    # Create a non-uniform stationary distribution (vector of length M)
    pi = torch.linspace(0.01, 1.0, M, device=device)
    pi = pi / pi.sum()
    
    # Compute F: (M x N) with each column computed in a batched manner.
    F = torch.empty((M, N), dtype=torch.float32, device=device)
    for n in range(N):
        F[:, n] = vectorized_eigenfunction(n, states)
    
    # Convert eigenvalues list to a tensor (shape: (N,))
    eigen_tensor = torch.tensor(eigenvalues, dtype=torch.float32, device=device)
    
    # Compute the weighted inner product:
    # For each i,j: P[i,j] = (sum_n F[i,n]*eigenvalues[n]*F[j,n]) * pi[j]
    B = F * eigen_tensor  # elementwise multiply each column by eigenvalues
    temp = F @ B.t()      # (M x M) matrix where entry (i,j) is the sum over n
    P = temp * pi.unsqueeze(0)  # Multiply each column j by pi[j]
    
    # Normalize rows so that each sums to 1.
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
    # Full state space (assumes |state_space| = |state_vals|^d)
    d = int(np.log2(P.shape[0]))
    full_states = get_state_space(state_vals, d)  # (num_full x d)
    
    # S_list: sorted list of indices to keep
    S_list = sorted(list(S))
    
    # If S_list is empty, reduced state space has one state (the empty tuple)
    if len(S_list) == 0:
        num_reduced = 1
        full_to_reduced = torch.zeros(full_states.shape[0], dtype=torch.long, device=device)
    else:
        # Extract the components corresponding to S_list
        full_reduced = full_states[:, S_list]  # (num_full x |S_list|)
        # Convert each row to a tuple of integers (0 or 1)
        full_reduced_tuples = [tuple(int(round(x)) for x in row) for row in full_reduced.tolist()]
        # Generate the reduced state space using itertools.product
        reduced_tuples = list(product(state_vals, repeat=len(S_list)))
        mapping = {r: i for i, r in enumerate(reduced_tuples)}
        # Map each full state's projection to its index in the reduced state space
        full_to_reduced = torch.tensor([mapping[r] for r in full_reduced_tuples],
                                        device=device, dtype=torch.long)
        num_reduced = len(reduced_tuples)
    
    # Compute the marginal distribution for the reduced state space.
    pi_full = compute_stationary_distribution(P)  # (num_full,)
    # Build a one-hot mapping matrix M (num_full x num_reduced)
    M = torch.zeros((full_states.shape[0], num_reduced), dtype=torch.float32, device=device)
    M[torch.arange(full_states.shape[0]), full_to_reduced] = 1.0
    pi_S = M.t() @ pi_full  # (num_reduced,)
    
    # Compute the reduced transition matrix:
    #    P_S = (M.T @ (diag(pi_full)*P) @ M) / pi_S
    weighted_P = (pi_full.unsqueeze(1) * P)  # (num_full x num_full)
    P_S_num = M.t() @ weighted_P @ M         # (num_reduced x num_reduced)
    
    # Normalize each row by its marginal probability.
    P_S = torch.zeros_like(P_S_num)
    for i in range(num_reduced):
        if pi_S[i] > 0:
            P_S[i, :] = P_S_num[i, :] / pi_S[i]
    return P_S

def compute_entropy_rate(P):
    """
    Computes the entropy rate H(P) in a vectorized manner.
    """
    pi = compute_stationary_distribution(P)  # (num_states,)
    # The entropy rate: H = - sum_i pi[i] * sum_j P[i,j] log P[i,j]
    entropy_rate = -torch.sum(pi.unsqueeze(1) * P * torch.log(P + 1e-10))
    return entropy_rate

def compute_shannon_entropy(P):
    return -torch.sum(P * torch.log(P + 1e-10))

def compute_joint_entropy(joint):
    return -torch.sum(joint * torch.log(joint + 1e-10))

# -----------------------
# Submodular Maximizer (remains in pure Python/NumPy)
# -----------------------
def greedy(f, X, k):
    """
    Greedy algorithm for submodular maximization with cardinality constraints.
    """
    S = set()
    plot_vals = []
    for _ in range(k):
        gains = [(f(S.union({e})) - f(S), e) for e in X - S]
        gain, elem = max(gains)
        S.add(elem)
        print(f"Added element {elem}")
        plot_vals.append(f(S))
    return plot_vals

def lazy_greedy(f, X, k):
    """
    Lazy greedy algorithm for submodular maximization with cardinality constraints.
    """
    S = set()
    gains = [(f(S.union({e})) - f(S), e) for e in X - S]
    gain, elem = max(gains)
    S.add(elem)
    for _ in range(k - 1):
        gains = [(f(S.union({e})) - f(S), e) for e in X - S]
        gain, elem = max(gains)
        S.add(elem)
    return S

def stochastic_greedy(f, X, k, epsilon=0.1):
    """
    Stochastic greedy algorithm for submodular maximization with cardinality constraints.
    """
    S = set()
    n = len(X)
    s = int((n / k) * np.log(1 / epsilon))
    for _ in range(k):
        R = np.random.choice(list(X - S), size=min(s, len(X - S)), replace=False)
        gains = [(f(S.union({e})) - f(S), e) for e in R]
        total = sum(g for g, _ in gains)
        probs = [g / total if total > 0 else 1/len(gains) for g, _ in gains]
        elem = np.random.choice([e for _, e in gains], p=probs)
        S.add(elem)
    return S

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
# Visualization (using NumPy & Matplotlib)
# -----------------------
def simulate_path(P, state_vals, steps, initial_state=None):
    """
    Simulate a path from the Markov chain with transition matrix P.
    P is a torch tensor (converted to CPU numpy array for simulation).
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
    """
    Plot sample paths from the original Markov chain and the reduced (subset) chain.
    """
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

# -----------------------
# Main Script
# -----------------------
if __name__ == "__main__":
    # Parameters for the full Markov chain
    N = 100
    d = 5
    state_vals = [0, 1]
    eigenvalues = [1 / (n + 1) for n in range(N)]
    
    # Define a GPU-enabled eigenfunction.
    # Note: This version expects a state x (a tuple or list) and returns a torch scalar.
    def eigenfunction(n, x, alpha=0.3):
        xt = torch.tensor(x, dtype=torch.float32, device=device)
        damping = torch.exp(-alpha * torch.sum(xt**2))
        periodic_part = torch.prod(1 + torch.cos((n + 1) * torch.pi * xt))
        return damping * periodic_part

    # Generate the full transition matrix P on the GPU.
    P = MC_generation(N, d, state_vals, eigenvalues)
    print(f"Generated multivariate reversible Markov chain with {d} dimensions.")
    
    # Define the submodular function.
    # We wrap the entropy rate computation so that it returns a Python float.
    submod_func = lambda S: compute_entropy_rate(keep_S_in_mat(P, state_vals, S)).item()
    X = set(range(d))
    m = d

    # Select a subset via lazy greedy (you can also try greedy or stochastic_greedy).
    #optimal_subset = lazy_greedy(submod_func, X, m)
    #print("Lazy greedy algorithm completed.")
    
    # Compute the reduced transition matrix and its entropy rate.
    #P_opt = keep_S_in_mat(P, state_vals, optimal_subset)
    #opt_entropy_rate = compute_entropy_rate(P_opt).item()
    
    #print(f"Optimal subset: {optimal_subset}")
    #print(f"Optimal matrix:\n{P_opt.cpu().numpy()}")
    #print(f"Optimal entropy rate: {opt_entropy_rate}")
    function_values = []
    function_values = greedy(submod_func, X, m)
    plot_objective_per_iteration(function_values)
    
    # -----------------------
    # Visualization
    # -----------------------
    #steps = 50
    #original_path = simulate_path(P, state_vals, steps)
    #subset_indices = sorted(optimal_subset)
    #subset_path = simulate_path(P_opt, state_vals, steps, initial_state=original_path[0, subset_indices])
    #plot_sample_paths(original_path, subset_path, list(optimal_subset))
    
    # Compare entropy rates for non-optimal subsets
    #non_optimal_subsets = [set(comb) for comb in combinations(range(d), k) if set(comb) != optimal_subset]
    #entropy_rates = {tuple(S): submod_func(S) for S in non_optimal_subsets}
    #print(f"Entropy rates of non-optimal subsets: {entropy_rates}")