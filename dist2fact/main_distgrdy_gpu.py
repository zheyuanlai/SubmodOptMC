#!/usr/bin/env python
"""
GPU-enabled distorted greedy submodular maximization approach.
Heavy linear-algebra computations (including KL divergence and outer products)
are performed with PyTorch on the GPU.
This script selects a subset of dimensions via a distorted greedy algorithm,
and computes the distance to factorizability using a KL divergence measure.
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
# GPU-ENABLED FUNCTIONS
# -----------------------

def get_state_space(state_vals, d):
    """
    Generates the full state space as a tensor of shape (num_states, d)
    using torch.cartesian_prod.
    """
    if d == 0:
        return torch.empty((1, 0), dtype=torch.float32, device=device)
    grids = [torch.tensor(state_vals, dtype=torch.float32, device=device) for _ in range(d)]
    return torch.cartesian_prod(*grids)

def vectorized_eigenfunction(n, states, alpha=0.3):
    """
    Computes the eigenfunction for a given n in a vectorized way.
    `states` is a tensor of shape (num_states, d).
    """
    damping = torch.exp(-alpha * torch.sum(states**2, dim=1))
    periodic = torch.prod(1 + torch.cos((n + 1) * torch.pi * states), dim=1)
    return damping * periodic

def MC_generation(N, d, state_vals, eigenvalues, eigenfunction):
    """
    Generates the transition matrix P for a reversible Markov chain.
    Instead of nested loops, we compute a feature matrix F (M x N) where
      F[i, n] = eigenfunction(n, state_i),
    and then form P via a weighted inner product.
    """
    states = get_state_space(state_vals, d)  # (M x d)
    M = states.shape[0]
    # Create a non-uniform stationary distribution vector pi
    pi = torch.linspace(0.01, 1.0, M, device=device)
    pi = pi / pi.sum()
    # Build feature matrix F (M x N)
    F = torch.empty((M, N), dtype=torch.float32, device=device)
    for n in range(N):
        F[:, n] = vectorized_eigenfunction(n, states)
    eigen_tensor = torch.tensor(eigenvalues, dtype=torch.float32, device=device)
    B = F * eigen_tensor  # scale columns
    temp = F @ B.t()      # (M x M) matrix with entries sum_n F[i,n]*eigenvalues[n]*F[j,n]
    P = temp * pi.unsqueeze(0)  # weight columns by pi[j]
    P = P / (P.sum(dim=1, keepdim=True) + 1e-10)  # row normalize
    return P

def compute_stationary_distribution(P):
    """
    Computes the stationary distribution for a reversible Markov chain P using detailed balance.
    Assumes: pi[i] = pi[i-1]*(P[i-1, i]/P[i, i-1]).
    """
    n = P.shape[0]
    pi_unnormalized = torch.ones(n, device=P.device, dtype=P.dtype)
    for i in range(1, n):
        pi_unnormalized[i] = pi_unnormalized[i-1] * (P[i-1, i] / (P[i, i-1] + 1e-10))
    return pi_unnormalized / torch.sum(pi_unnormalized)

def keep_S_in_mat(P, state_vals, S):
    """
    Computes the reduced transition matrix P_S on the subset of dimensions S.
    The full state space is assumed to be product(state_vals, repeat=d),
    where d = log2(|P|).
    """
    d = int(np.log2(P.shape[0]))
    full_states = list(product(state_vals, repeat=d))
    S_list = sorted(list(S))
    if len(S_list) == 0:
        num_reduced = 1
        full_to_reduced = torch.zeros(len(full_states), dtype=torch.long, device=device)
    else:
        full_reduced = [tuple(x[k] for k in S_list) for x in full_states]
        reduced_states = list(product(state_vals, repeat=len(S_list)))
        mapping = {state: i for i, state in enumerate(reduced_states)}
        full_to_reduced = torch.tensor([mapping[r] for r in full_reduced],
                                        device=device, dtype=torch.long)
        num_reduced = len(reduced_states)
    pi_full = compute_stationary_distribution(P)
    M_onehot = torch.zeros((len(full_states), num_reduced), dtype=torch.float32, device=device)
    M_onehot[torch.arange(len(full_states)), full_to_reduced] = 1.0
    pi_S = M_onehot.t() @ pi_full
    weighted_P = (pi_full.unsqueeze(1) * P)
    P_S_num = M_onehot.t() @ weighted_P @ M_onehot
    P_S = torch.zeros_like(P_S_num)
    for i in range(num_reduced):
        if pi_S[i] > 0:
            P_S[i, :] = P_S_num[i, :] / pi_S[i]
    return P_S

def leave_S_out_mat(P, state_vals, S):
    """
    Computes the leave-S-out matrix P_{-S} by excluding the dimensions in S.
    This is done by calling keep_S_in_mat with the complementary subset.
    """
    d = int(np.log2(P.shape[0]))
    minus_S = {i for i in range(d) if i not in S}
    return keep_S_in_mat(P, state_vals, minus_S)

def compute_entropy_rate(P):
    """
    Computes the entropy rate H(P) = - sum_i pi[i] * sum_j P[i,j] log(P[i,j]),
    where pi is the stationary distribution.
    """
    pi = compute_stationary_distribution(P)
    return -torch.sum(pi.unsqueeze(1) * P * torch.log(P + 1e-10))

# -----------------------
# GPU VERSIONS OF KL DIVERGENCE AND OUTER PRODUCT
# -----------------------

def KL_divergence_gpu(P, Q):
    """
    Computes the Kullback-Leibler divergence KL(P||Q) on the GPU.
    P and Q are torch tensors (transition matrices).
    KL(P||Q) = sum_x pi[x] * sum_y P[x,y] * log(P[x,y]/Q[x,y]),
    where pi is the stationary distribution of P.
    """
    pi = compute_stationary_distribution(P)
    kl = torch.sum(pi.unsqueeze(1) * P * torch.log((P + 1e-10)/(Q + 1e-10)))
    return kl

def compute_outer_product_gpu(A, B):
    """
    Computes the outer (Kronecker) product of matrices A and B on the GPU.
    Uses torch.kron for efficiency.
    """
    return torch.kron(A, B)

# -----------------------
# DISTORTED GREEDY ALGORITHM (PURE PYTHON)
# -----------------------

def distorted_greedy(f, c, U, m):
    """
    Implements the corrected distorted greedy algorithm.
    
    Parameters:
      f : function
          A monotonic non-decreasing submodular function.
      c : function
          A modular cost function.
      U : set
          The ground set of elements.
      m : int
          The cardinality constraint.
    
    Returns:
      set: The selected subset.
    """
    S = set()
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
    return S

# -----------------------
# VISUALIZATION FUNCTIONS
# -----------------------

def simulate_path(P, state_vals, steps, initial_state=None):
    """
    Simulates a path from the Markov chain with transition matrix P.
    P is a torch tensor and is converted to a CPU NumPy array for simulation.
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
    Plots sample paths from the original Markov chain and the reduced (subset) chain.
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
# MAIN SCRIPT
# -----------------------

if __name__ == "__main__":
    # Parameters for the full Markov chain
    N = 100
    d = 5
    state_vals = [0, 1]
    eigenvalues = [1 / (n + 1) for n in range(N)]
    
    # Define a GPU-enabled eigenfunction.
    def eigenfunction(n, x, alpha=0.3):
        xt = torch.tensor(x, dtype=torch.float32, device=device)
        damping = torch.exp(-alpha * torch.sum(xt**2))
        periodic_part = torch.prod(1 + torch.cos((n + 1) * torch.pi * xt))
        return damping * periodic_part

    # Generate the full transition matrix P on the GPU.
    P = MC_generation(N, d, state_vals, eigenvalues, eigenfunction)
    print(f"Generated multivariate reversible Markov chain with {d} dimensions.")
    
    # Define cost function c and submodular function g using GPU versions of KL divergence and outer product.
    def c(S):
        total = 0.0
        for elem in S:
            P_S_single = keep_S_in_mat(P, state_vals, {elem})
            P_minus_S_single = leave_S_out_mat(P, state_vals, {elem})
            outer_mat = compute_outer_product_gpu(P_S_single, P_minus_S_single)
            total += KL_divergence_gpu(P, outer_mat).item()
        return total

    def g(S):
        P_S = keep_S_in_mat(P, state_vals, S)
        P_minus_S = leave_S_out_mat(P, state_vals, S)
        outer_mat = compute_outer_product_gpu(P_S, P_minus_S)
        return KL_divergence_gpu(P, outer_mat).item() + c(S)
    
    X = set(range(d))
    m = 2

    # Run the distorted greedy algorithm.
    optimal_subset = distorted_greedy(g, c, X, m)
    print("Distorted greedy algorithm completed.")
    
    # Compute the reduced transition matrix for the optimal subset.
    P_opt = keep_S_in_mat(P, state_vals, optimal_subset)
    P_minus_opt = leave_S_out_mat(P, state_vals, optimal_subset)
    outer_mat = compute_outer_product_gpu(P_opt, P_minus_opt)
    dist2fact = KL_divergence_gpu(P, outer_mat).item()
    
    # Visualization: simulate paths for full and reduced chains.
    #steps = 50
    #original_path = simulate_path(P, state_vals, steps)
    #subset_indices = sorted(optimal_subset)
    #subset_path = simulate_path(P_opt, state_vals, steps, initial_state=original_path[0, subset_indices])
    #plot_sample_paths(original_path, subset_path, list(optimal_subset))
    
    print(f"Optimal subset: {optimal_subset}")
    print(f"Optimal matrix (P_S):\n{P_opt.cpu().numpy()}")
    print(f"Distance to factorizability: {dist2fact}")
    
    # Compute distances for non-optimal subsets.
    non_optimal_subsets = [set(comb) for comb in combinations(range(d), m) if set(comb) != optimal_subset]
    dist2facts = {tuple(S): KL_divergence_gpu(
                    P,
                    compute_outer_product_gpu(
                        keep_S_in_mat(P, state_vals, S),
                        leave_S_out_mat(P, state_vals, S)
                    )
                  ).item() for S in non_optimal_subsets}
    print(f"Distances to factorizability of non-optimal subsets: {dist2facts}")
