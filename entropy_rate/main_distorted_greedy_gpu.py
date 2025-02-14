#!/usr/bin/env python
"""
Combined GPU-enabled version of the original code.
Uses PyTorch to move the heavy linear‐algebra operations to the GPU
(with CUDA or MPS, if available).
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

# ======== mc_generation.py ========
def MC_generation(N, d, state_vals, eigenvalues, eigenfunction):
    """
    Generates a reversible Markov chain with N eigenfunctions and d dimensions.
    
    Parameters:
        N (int): The number of eigenfunctions.
        d (int): The number of dimensions.
        state_vals (list): The values of the state space.
        eigenvalues (list): The eigenvalues.
        eigenfunction (function): The corresponding eigenfunction.
        
    Returns:
        torch.Tensor: The transition matrix P (state_space_size x state_space_size) on device.
    """
    state_space = list(product(state_vals, repeat=d))
    state_space_size = len(state_space)
    # Create a non-uniform stationary distribution (as a torch tensor)
    pi = torch.linspace(0.01, 1.0, state_space_size, device=device)
    pi = pi / pi.sum()
    # Allocate transition matrix P on device
    P = torch.zeros((state_space_size, state_space_size), dtype=torch.float32, device=device)
    for i, x in enumerate(state_space):
        for j, y in enumerate(state_space):
            s = 0.0
            for n in range(N):
                # eigenfunction is expected to return a torch scalar on device.
                s += eigenvalues[n] * eigenfunction(n, x) * eigenfunction(n, y) * pi[j]
            P[i, j] = s
    # Normalize rows of P
    P = P / P.sum(dim=1, keepdim=True)
    return P

def compute_stationary_distribution(P):
    """
    Computes the stationary distribution of a Markov chain with transition matrix P.
    
    Parameters:
        P (torch.Tensor): Transition matrix (on device).
        
    Returns:
        torch.Tensor: Stationary distribution (1D tensor on device).
    """
    # Transpose P because we want right eigenvector of P.T corresponding to eigenvalue 1
    eigvals, eigvecs = torch.linalg.eig(P.t())
    # Find index of eigenvalue approximately equal to 1 (using real part)
    tol = 1e-5
    one = torch.tensor(1.0, device=device)
    idx = torch.nonzero(torch.isclose(eigvals.real, one, atol=tol))
    if idx.numel() == 0:
        raise ValueError("No eigenvalue equal to 1 found.")
    # Take the first matching eigenvector and use its real part
    stationary_vector = eigvecs[:, idx[0].item()].real
    stationary_distribution = stationary_vector / stationary_vector.sum()
    return stationary_distribution

# ======== keep_S_in.py ========
def marginal_pi(S, P, state_space, reduced_state_space):
    """
    Computes the marginal distribution pi_S over the reduced state space.
    
    Parameters:
        S (set): Subset (indices of dimensions) used.
        P (torch.Tensor): Full transition matrix.
        state_space (list): Full state space (list of tuples).
        reduced_state_space (list): Reduced state space (list of tuples).
    
    Returns:
        torch.Tensor: Marginal distribution pi_S (on device).
    """
    pi = compute_stationary_distribution(P)
    marg_dist = torch.zeros(len(reduced_state_space), dtype=torch.float32, device=device)
    reduced_state_map = {state: idx for idx, state in enumerate(reduced_state_space)}
    for i, state in enumerate(state_space):
        partial_state = tuple(state[j] for j in S)
        partial_index = reduced_state_map[partial_state]
        marg_dist[partial_index] += pi[i]
    return marg_dist

def keep_S_in_mat(P, state_vals, S):
    """
    Computes the keep-S-in matrix P_S on the subset of state space (S).
    
    Parameters:
        P (torch.Tensor): Transition matrix (on device).
        state_vals (list): The values of the state space.
        S (set): The subset of dimensions (indices) to keep.
    
    Returns:
        torch.Tensor: The reduced transition matrix P_S (on device).
    """
    # Compute full state space (assumes binary states and d = log2(|state_space|))
    d = int(np.log2(P.shape[0]))
    state_space = list(product(state_vals, repeat=d))
    reduced_state_space = list(product(state_vals, repeat=len(S)))
    # Compute marginal distribution for the reduced state space
    pi_S = marginal_pi(S, P, state_space, reduced_state_space)
    P_S = torch.zeros((len(pi_S), len(pi_S)), dtype=torch.float32, device=device)
    
    # Accumulate weighted transitions
    reduced_state_map = {state: idx for idx, state in enumerate(reduced_state_space)}
    full_pi = compute_stationary_distribution(P)
    for i, x in enumerate(state_space):
        for j, y in enumerate(state_space):
            # Map full states x and y to their reduced versions
            x_S = tuple(x[k] for k in S)
            y_S = tuple(y[k] for k in S)
            x_S_index = reduced_state_map[x_S]
            y_S_index = reduced_state_map[y_S]
            P_S[x_S_index, y_S_index] += full_pi[i] * P[i, j]
    # Normalize rows of P_S by the marginal probability
    for i in range(len(pi_S)):
        if pi_S[i] > 0:
            P_S[i, :] /= pi_S[i]
    return P_S

# ======== entropy_rate.py ========
def compute_entropy_rate(P):
    """
    Computes the entropy rate H(P) of the transition probability matrix P of a Markov chain.
    
    Parameters:
        P (torch.Tensor): Transition matrix (on device).
    
    Returns:
        torch.Tensor: Entropy rate (a scalar tensor).
    """
    pi = compute_stationary_distribution(P)
    entropy_rate = 0.0
    # Use nested loops over states (for moderate state space sizes)
    for i, pi_x in enumerate(pi):
        for j, P_xy in enumerate(P[i]):
            if P_xy > 0:
                entropy_rate -= pi_x * P_xy * torch.log(P_xy)
    return entropy_rate

def compute_shannon_entropy(P):
    """
    Computes the Shannon entropy of a distribution (or matrix).
    
    Parameters:
        P (torch.Tensor): Tensor (on device).
    
    Returns:
        torch.Tensor: Shannon entropy.
    """
    return -torch.sum(P * torch.log(P + 1e-10))

def compute_joint_entropy(joint):
    """
    Computes the joint entropy given a joint distribution.
    
    Parameters:
        joint (torch.Tensor): Joint distribution tensor (on device).
    
    Returns:
        torch.Tensor: Joint entropy.
    """
    return -torch.sum(joint * torch.log(joint + 1e-10))

# ======== distorted_greedy.py ========
def distorted_greedy(f, c, U, m):
    """
    Implements the corrected Distorted Greedy algorithm.
    
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
        set: The selected subset S.
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
        # Only add the element if the computed gain is positive.
        if (1 - 1/m) ** (m - (i + 1)) * (f(S | {best_e}) - f(S)) - c({best_e}) > 0:
            S.add(best_e)
    return S

# ======== visualization.py ========
def simulate_path(P, state_vals, steps, initial_state=None):
    """
    Simulates a path from a Markov chain defined by transition matrix P.
    
    Parameters:
        P (torch.Tensor): Transition matrix (on device).
        state_vals (list): Values for each state.
        steps (int): Number of steps to simulate.
        initial_state (array-like): Optional initial state.
    
    Returns:
        np.ndarray: Array of states along the simulated path.
    """
    # Determine d from the size of P (assuming binary states, so size = 2^d)
    d = int(np.log2(P.shape[0]))
    
    def state_to_index(state):
        return int("".join(map(str, state)), 2)
    
    def index_to_state(index):
        return list(map(int, f"{index:0{d}b}"))
    
    if initial_state is None:
        initial_state = np.random.choice(state_vals, size=d)
    
    current_state = initial_state
    states = [current_state]
    # Convert P to CPU numpy array for simulation and random choice
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
    Plots and saves sample paths of the original Markov chain and the subset chain.
    
    Parameters:
        original_path (np.ndarray): Path of the full Markov chain.
        subset_path (np.ndarray): Path of the reduced (subset) chain.
        subset_indices (list): The indices of dimensions in the subset.
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

# ======== Main script (main_distorted_greedy.py) ========
if __name__ == "__main__":
    # Parameters for the full Markov chain
    N = 100
    d = 5
    state_vals = [0, 1]
    eigenvalues = [1 / (n + 1) for n in range(N)]
    
    # Define the eigenfunction using torch operations so that computations occur on the device.
    def eigenfunction(n, x, alpha=0.3):
        # Convert the tuple x to a torch tensor (float32) on the chosen device.
        xt = torch.tensor(x, dtype=torch.float32, device=device)
        damping = torch.exp(-torch.sum(alpha * xt**2))
        # Compute the periodic part: product over dimensions of [1 + cos((n+1)*pi*xi)]
        periodic_part = torch.prod(1 + torch.cos((n + 1) * torch.pi * xt))
        return damping * periodic_part

    # Generate the full Markov chain transition matrix on the device.
    P = MC_generation(N, d, state_vals, eigenvalues, eigenfunction)
    print(f"Generated multivariate reversible Markov chain with {d} dimensions.")
    
    # Define the submodular function f and the modular cost function c.
    def submod_func(S):
        P_S = keep_S_in_mat(P, state_vals, S)
        pi_S = compute_stationary_distribution(P_S)
        return compute_joint_entropy(pi_S[:, None] * P_S)
    
    def modular_func(S):
        P_S = keep_S_in_mat(P, state_vals, S)
        pi_S = compute_stationary_distribution(P_S)
        return compute_shannon_entropy(pi_S)
    
    X = set(range(d))
    m = 3
    
    # Run the Distorted Greedy algorithm to select a subset.
    optimal_subset = distorted_greedy(submod_func, modular_func, X, m)
    print("Distorted greedy algorithm completed.")
    
    # Compute the entropy rate of the chain defined on the optimal subset.
    P_opt = keep_S_in_mat(P, state_vals, optimal_subset)
    opt_entropy_rate = compute_entropy_rate(P_opt)
    
    print(f"Optimal subset: {optimal_subset}")
    print(f"Optimal matrix (P_opt):\n{P_opt.cpu().numpy()}")
    print(f"Optimal entropy rate: {opt_entropy_rate.item()}")
    
    # Visualization: simulate paths from the original and the reduced (subset) chains.
    #steps = 50
    #original_path = simulate_path(P, state_vals, steps)
    #subset_indices = sorted(optimal_subset)
    # Use the initial state for the subset chain from the corresponding dimensions of the full chain’s first state.
    #initial_subset_state = original_path[0, subset_indices]
    #subset_path = simulate_path(P_opt, state_vals, steps, initial_state=initial_subset_state)
    #plot_sample_paths(original_path, subset_path, list(optimal_subset))
    
    # Compute and print entropy rates for all non-optimal subsets of size m.
    non_optimal_subsets = [
        set(comb) for comb in combinations(range(d), m)
        if set(comb) != optimal_subset
    ]
    entropy_rates = {}
    for S in non_optimal_subsets:
        entropy = compute_entropy_rate(keep_S_in_mat(P, state_vals, S))
        # Convert tensor to Python float
        entropy_rates[tuple(sorted(S))] = entropy.item()
    print(f"Entropy rates of non-optimal subsets: {entropy_rates}")
