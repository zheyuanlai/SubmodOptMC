#!/usr/bin/env python3
"""
MCMC tensor product comparison experiment on the Curie-Weiss model (d=8).
This script compares:
(i) P^{n*} - the original dynamics raised to power n*
(ii) (P^{(-4)})^{n*} ⊗ (P^{(4)})^{n*} - the tensor product of marginal chains

Where n* is the time when d_{TV}^{(4)}(n*) first falls below 0.2 (estimated at n*=10).

We compare these by:
1. Computing operator TV norms: ||P^{n*} - Π||_{TV} vs ||(P^{(-4)} ⊗ P^{(4)})^{n*} - Π||_{TV}
2. Simulating samples from both dynamics and computing empirical TV distances to π
"""

import math
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def build_state_space(d):
    """Build the full state space {-1,+1}^d."""
    states = np.array(list(product([-1, 1], repeat=d)), dtype=np.int8)
    return states


def build_weight_matrix(d):
    """Build the weight matrix W_ij = 2^{-|i-j|}."""
    idx = np.arange(d)
    diff = np.abs(idx[:, None] - idx[None, :])
    return 2.0 ** (-diff)


def hamiltonian(states, weights, h):
    """Compute the Hamiltonian H(x) = -x^T W x - h * sum(x)."""
    interaction = np.einsum("ij,ij->i", states @ weights, states)
    field = h * np.sum(states, axis=1)
    return -interaction - field


def stationary_distribution(energies, beta):
    """Compute the stationary distribution π(x) ∝ exp(-β H(x))."""
    unnorm = np.exp(-beta * energies)
    return unnorm / np.sum(unnorm)


def compute_transition_matrix(states, energies, beta):
    """Compute the Metropolis transition matrix P."""
    m, d = states.shape
    state_to_index = {tuple(state.tolist()): i for i, state in enumerate(states)}
    P = np.zeros((m, m), dtype=np.float64)
    
    for i, x in enumerate(states):
        total_rate = 0.0
        for j in range(d):
            y = x.copy()
            y[j] = -y[j]
            k = state_to_index[tuple(y.tolist())]
            delta = energies[k] - energies[i]
            acc = math.exp(-beta * max(delta, 0.0))
            rate = (1.0 / d) * acc
            P[i, k] = rate
            total_rate += rate
        P[i, i] = 1.0 - total_rate
    
    return P


def aggregate_chain(P, pi, states, keep_indices):
    """
    Aggregate the chain to a marginal chain on keep_indices.
    Returns: reduced_states, pi_marginal, P_marginal
    """
    keep = sorted(keep_indices)
    m = states.shape[0]
    
    # Build mapping from full states to reduced states
    reduced_map = {}
    reduced_states = []
    full_to_reduced = np.empty(m, dtype=np.int32)
    
    for i, x in enumerate(states):
        key = tuple(x[keep])
        if key not in reduced_map:
            reduced_map[key] = len(reduced_map)
            reduced_states.append(key)
        full_to_reduced[i] = reduced_map[key]
    
    r = len(reduced_states)
    
    # Compute marginal distribution
    pi_r = np.zeros(r, dtype=np.float64)
    for i in range(m):
        pi_r[full_to_reduced[i]] += pi[i]
    
    # Compute marginal transition matrix
    P_num = np.zeros((r, r), dtype=np.float64)
    for i in range(m):
        ri = full_to_reduced[i]
        for j in range(m):
            rj = full_to_reduced[j]
            P_num[ri, rj] += pi[i] * P[i, j]
    
    P_r = np.zeros_like(P_num)
    for i in range(r):
        if pi_r[i] > 0:
            P_r[i, :] = P_num[i, :] / pi_r[i]
    
    return reduced_states, pi_r, P_r


def tensor_product_transition(P_minus_i, P_i, states, removed_idx):
    """
    Construct the tensor product transition matrix (P^{(-i)})^n ⊗ (P^{(i)})^n.
    
    P_minus_i: transition matrix for the marginal chain excluding coordinate removed_idx
    P_i: transition matrix for the marginal chain on coordinate removed_idx only
    states: full state space
    removed_idx: the coordinate that was removed (0-indexed)
    
    Returns: transition matrix on the full state space
    """
    m = states.shape[0]
    d = states.shape[1]
    
    # Build mappings
    state_to_index = {tuple(state.tolist()): i for i, state in enumerate(states)}
    
    # Build mapping for marginal states (excluding removed_idx)
    keep_indices = [j for j in range(d) if j != removed_idx]
    marginal_map = {}
    for i, x in enumerate(states):
        key = tuple(x[keep_indices])
        if key not in marginal_map:
            marginal_map[key] = len(marginal_map)
    
    # Build mapping for removed coordinate
    single_map = {-1: 0, 1: 1}
    
    # Construct tensor product
    P_tensor = np.zeros((m, m), dtype=np.float64)
    
    for i, x in enumerate(states):
        # Get indices in marginal spaces
        marginal_key = tuple(x[keep_indices])
        marginal_idx = marginal_map[marginal_key]
        single_idx = single_map[x[removed_idx]]
        
        for j, y in enumerate(states):
            marginal_key_y = tuple(y[keep_indices])
            marginal_idx_y = marginal_map[marginal_key_y]
            single_idx_y = single_map[y[removed_idx]]
            
            # Tensor product: P_{ij} = P_marginal[idx_x_marginal, idx_y_marginal] * P_single[idx_x_single, idx_y_single]
            P_tensor[i, j] = P_minus_i[marginal_idx, marginal_idx_y] * P_i[single_idx, single_idx_y]
    
    return P_tensor


def operator_tv_norm(P, pi):
    """
    Compute ||P - Π||_{TV} where Π is the stationary distribution matrix.
    ||P - Π||_{TV} = max_i sum_j |P_{ij} - π_j| / 2
    """
    m = P.shape[0]
    pi_matrix = np.tile(pi, (m, 1))
    diff = np.abs(P - pi_matrix)
    return 0.5 * np.max(np.sum(diff, axis=1))


def simulate_markov_chain(P, initial_state_idx, n_steps):
    """
    Simulate a Markov chain for n_steps starting from initial_state_idx.
    Returns the final state index.
    """
    current_state = initial_state_idx
    for _ in range(n_steps):
        # Sample next state according to transition probabilities
        current_state = np.random.choice(P.shape[0], p=P[current_state, :])
    return current_state


def simulate_tensor_chain(P_minus_i, P_i, states, removed_idx, initial_state_idx, n_steps):
    """
    Simulate the tensor product chain (P^{(-i)})^n ⊗ (P^{(i)})^n.
    
    This simulates:
    1. P^{(-i)} for n_steps starting from the marginal of initial_state
    2. P^{(i)} for n_steps starting from the removed coordinate of initial_state
    3. Concatenates the results to get the full state
    """
    d = states.shape[1]
    keep_indices = [j for j in range(d) if j != removed_idx]
    
    initial_state = states[initial_state_idx]
    
    # Build marginal state spaces
    marginal_states_list = []
    marginal_map = {}
    for i, x in enumerate(states):
        key = tuple(x[keep_indices])
        if key not in marginal_map:
            marginal_map[key] = len(marginal_states_list)
            marginal_states_list.append(list(key))
    marginal_states = np.array(marginal_states_list, dtype=np.int8)
    
    single_states = np.array([[-1], [1]], dtype=np.int8)
    single_map = {-1: 0, 1: 1}
    
    # Get initial states in marginal spaces
    marginal_initial_key = tuple(initial_state[keep_indices])
    marginal_initial_idx = marginal_map[marginal_initial_key]
    single_initial_idx = single_map[initial_state[removed_idx]]
    
    # Simulate marginal chain
    marginal_final_idx = simulate_markov_chain(P_minus_i, marginal_initial_idx, n_steps)
    marginal_final = marginal_states[marginal_final_idx]
    
    # Simulate single coordinate chain
    single_final_idx = simulate_markov_chain(P_i, single_initial_idx, n_steps)
    single_final = single_states[single_final_idx, 0]
    
    # Concatenate results
    final_state = np.zeros(d, dtype=np.int8)
    for idx, j in enumerate(keep_indices):
        final_state[j] = marginal_final[idx]
    final_state[removed_idx] = single_final
    
    # Find the index of the final state
    state_to_index = {tuple(state.tolist()): i for i, state in enumerate(states)}
    final_state_idx = state_to_index[tuple(final_state.tolist())]
    
    return final_state_idx


def empirical_tv_distance(samples, pi):
    """
    Compute the TV distance between empirical distribution of samples and π.
    """
    m = len(pi)
    empirical = np.bincount(samples, minlength=m) / len(samples)
    return 0.5 * np.sum(np.abs(empirical - pi))


def main():
    # Parameters
    d = 8
    beta = 0.1
    h = 1.0
    n_star = 10  # Time when d_{TV}^{(4)}(n*) first falls below 0.2
    n_samples = 1000  # Number of samples to generate
    removed_idx = 3  # Remove coordinate 4 (0-indexed: 3)
    seeds = [100, 150, 200, 300]  # Four different seeds to test robustness
    
    print("=" * 70)
    print("MCMC Tensor Product Comparison Experiment (Multiple Seeds)")
    print("Curie-Weiss Model (d=8)")
    print("=" * 70)
    print(f"Parameters: β={beta}, h={h}, d={d}")
    print(f"Time steps: n*={n_star}")
    print(f"Removed coordinate: {removed_idx + 1} (1-indexed)")
    print(f"Number of samples: {n_samples}")
    print(f"Random seeds: {seeds}")
    print()
    
    # Build state space and dynamics
    states = build_state_space(d)
    weights = build_weight_matrix(d)
    energies = hamiltonian(states, weights, h)
    pi = stationary_distribution(energies, beta)
    P = compute_transition_matrix(states, energies, beta)
    
    m = states.shape[0]
    print(f"Full state space size: {m}")
    
    # Compute marginal chains
    keep_indices = [j for j in range(d) if j != removed_idx]
    print(f"Computing marginal chain P^{{(-{removed_idx + 1})}}...")
    _, pi_minus_i, P_minus_i = aggregate_chain(P, pi, states, keep_indices)
    
    print(f"Computing marginal chain P^{{({removed_idx + 1})}}...")
    _, pi_i, P_i = aggregate_chain(P, pi, states, [removed_idx])
    
    print(f"Marginal state space size (excluding coord {removed_idx + 1}): {P_minus_i.shape[0]}")
    print(f"Single coordinate state space size: {P_i.shape[0]}")
    print()
    
    # ====================================================================
    # Method 1: Operator TV Norm Comparison
    # ====================================================================
    print("-" * 70)
    print("Method 1: Operator TV Norm Comparison")
    print("-" * 70)
    
    # (i) Compute P^{n*}
    print(f"Computing P^{{{n_star}}}...")
    P_n = np.linalg.matrix_power(P, n_star)
    tv_original = operator_tv_norm(P_n, pi)
    print(f"||P^{{{n_star}}} - Π||_{{TV}} = {tv_original:.6f}")
    
    # (ii) Compute (P^{(-4)})^{n*} ⊗ (P^{(4)})^{n*}
    print(f"Computing (P^{{(-{removed_idx + 1})}})^{{{n_star}}}...")
    P_minus_i_n = np.linalg.matrix_power(P_minus_i, n_star)
    
    print(f"Computing (P^{{({removed_idx + 1})}})^{{{n_star}}}...")
    P_i_n = np.linalg.matrix_power(P_i, n_star)
    
    print(f"Computing tensor product...")
    P_tensor_n = tensor_product_transition(P_minus_i_n, P_i_n, states, removed_idx)
    tv_tensor = operator_tv_norm(P_tensor_n, pi)
    print(f"||(P^{{(-{removed_idx + 1})}})^{{{n_star}}} ⊗ (P^{{({removed_idx + 1})}})^{{{n_star}}} - Π||_{{TV}} = {tv_tensor:.6f}")
    
    print()
    print(f"Ratio (Tensor / Original): {tv_tensor / tv_original:.4f}")
    if tv_tensor < tv_original:
        print("✓ Tensor product has SMALLER TV distance!")
    else:
        print("✗ Tensor product has LARGER TV distance.")
    print()
    
    # ====================================================================
    # Method 2: Sample-based TV Distance Comparison (Multiple Seeds)
    # ====================================================================
    print("-" * 70)
    print("Method 2: Sample-based TV Distance Comparison (Multiple Seeds)")
    print("-" * 70)
    
    # Choose a common initial state (e.g., the all +1 state)
    initial_state = np.ones(d, dtype=np.int8)
    state_to_index = {tuple(state.tolist()): i for i, state in enumerate(states)}
    initial_state_idx = state_to_index[tuple(initial_state.tolist())]
    print(f"Initial state: {initial_state.tolist()}")
    print()
    
    # Store results for all seeds
    all_samples_original = []
    all_samples_tensor = []
    all_tv_original = []
    all_tv_tensor = []
    
    for seed_idx, seed in enumerate(seeds):
        print(f"--- Seed {seed} ({seed_idx + 1}/{len(seeds)}) ---")
        np.random.seed(seed)
        
        # (i) Simulate P^{n*}
        print(f"Simulating {n_samples} samples from P^{{{n_star}}}...")
        samples_original = []
        for _ in range(n_samples):
            final_state_idx = simulate_markov_chain(P, initial_state_idx, n_star)
            samples_original.append(final_state_idx)
        samples_original = np.array(samples_original)
        tv_samples_original = empirical_tv_distance(samples_original, pi)
        print(f"Empirical TV distance (P^{{{n_star}}} samples vs π): {tv_samples_original:.6f}")
        
        # (ii) Simulate (P^{(-4)})^{n*} ⊗ (P^{(4)})^{n*}
        print(f"Simulating {n_samples} samples from tensor product...")
        samples_tensor = []
        for _ in range(n_samples):
            final_state_idx = simulate_tensor_chain(P_minus_i, P_i, states, removed_idx, 
                                                    initial_state_idx, n_star)
            samples_tensor.append(final_state_idx)
        samples_tensor = np.array(samples_tensor)
        tv_samples_tensor = empirical_tv_distance(samples_tensor, pi)
        print(f"Empirical TV distance (Tensor samples vs π): {tv_samples_tensor:.6f}")
        
        print(f"Ratio (Tensor / Original): {tv_samples_tensor / tv_samples_original:.4f}")
        if tv_samples_tensor < tv_samples_original:
            print("✓ Tensor product samples have SMALLER TV distance!")
        else:
            print("✗ Tensor product samples have LARGER TV distance.")
        print()
        
        # Store results
        all_samples_original.append(samples_original)
        all_samples_tensor.append(samples_tensor)
        all_tv_original.append(tv_samples_original)
        all_tv_tensor.append(tv_samples_tensor)
    
    # ====================================================================
    # Plotting
    # ====================================================================
    print("-" * 70)
    print("Generating CDF plots for all seeds...")
    print("-" * 70)
    
    # Create figure with subplots for each seed
    output_path = Path(__file__).resolve().parent / "cw_tensor_comparison_cdf_multi.png"
    fig, axes = plt.subplots(1, 4 , figsize=(18, 5))
    
    state_indices = np.arange(m)
    cdf_pi = np.cumsum(pi[state_indices])
    
    for idx, (seed, samples_original, samples_tensor) in enumerate(zip(seeds, all_samples_original, all_samples_tensor)):
        ax = axes[idx]
        
        # Compute empirical distributions
        empirical_original = np.bincount(samples_original, minlength=m) / len(samples_original)
        empirical_tensor = np.bincount(samples_tensor, minlength=m) / len(samples_tensor)
        
        # Compute CDFs
        cdf_original = np.cumsum(empirical_original[state_indices])
        cdf_tensor = np.cumsum(empirical_tensor[state_indices])
        
        ax.plot(state_indices, cdf_pi, '-', label='π (Stationary)', linewidth=2.5, color='black', alpha=0.9)
        ax.plot(state_indices, cdf_original, '--', label=f'P^{10} samples', linewidth=2, alpha=0.8, color='blue')
        ax.plot(state_indices, cdf_tensor, '-.', label=f'Tensor samples', linewidth=2, alpha=0.8, color='red')
        
        ax.set_xlabel('State Index', fontsize=11)
        ax.set_ylabel('Cumulative Probability', fontsize=11)
        ax.set_title(f'Seed {seed}\nTV dist.: P^{10}={all_tv_original[idx]:.4f}, Tensor={all_tv_tensor[idx]:.4f}', 
                     fontsize=11)
        ax.legend(fontsize=9, loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, m])
        ax.set_ylim([0, 1])
    
    plt.suptitle(f'Empirical CDF Comparison', fontsize=14)
    plt.tight_layout()
    
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"Saved multi-seed CDF plot to {output_path}")
    print()
    
    # ====================================================================
    # Summary
    # ====================================================================
    print("=" * 70)
    print("SUMMARY (Across All Seeds)")
    print("=" * 70)
    print(f"Method 1 (Operator Norm):")
    print(f"  Original: {tv_original:.6f}")
    print(f"  Tensor:   {tv_tensor:.6f}")
    print(f"  Improvement: {(1 - tv_tensor / tv_original) * 100:.2f}%")
    print()
    print(f"Method 2 (Empirical Samples):")
    for idx, seed in enumerate(seeds):
        print(f"  Seed {seed}:")
        print(f"    Original: {all_tv_original[idx]:.6f}")
        print(f"    Tensor:   {all_tv_tensor[idx]:.6f}")
        print(f"    Improvement: {(1 - all_tv_tensor[idx] / all_tv_original[idx]) * 100:.2f}%")
    print()
    avg_tv_original = np.mean(all_tv_original)
    avg_tv_tensor = np.mean(all_tv_tensor)
    print(f"  Average across seeds:")
    print(f"    Original: {avg_tv_original:.6f}")
    print(f"    Tensor:   {avg_tv_tensor:.6f}")
    print(f"    Average Improvement: {(1 - avg_tv_tensor / avg_tv_original) * 100:.2f}%")
    print()
    
    # Check consistency
    improvement_count = sum(1 for i in range(len(seeds)) if all_tv_tensor[i] < all_tv_original[i])
    print(f"Tensor product showed improvement in {improvement_count}/{len(seeds)} seeds")
    
    if improvement_count == len(seeds):
        print("✓✓ Tensor product CONSISTENTLY shows improvement across all seeds!")
    elif improvement_count > 0:
        print("✓ Tensor product shows improvement in some seeds.")
    else:
        print("Note: No consistent improvement observed.")
    print("=" * 70)


if __name__ == "__main__":
    main()
