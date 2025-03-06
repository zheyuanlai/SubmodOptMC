#!/usr/bin/env python
"""
Optimized GPU-enabled version for the 10-dimensional Curie–Weiss model.
State space: all vectors in {-1,+1}^d.
We construct the Glauber dynamics transition matrix for the Curie–Weiss Hamiltonian
    H(x) = - (sum_i x_i)^2 - h * (sum_i x_i),
and the stationary (Gibbs) distribution
    pi(x) = exp(-beta H(x)) / Z.
An aggregation of states is performed by projecting onto a subset S of coordinates.
A greedy algorithm is used to select S to maximize the entropy rate of the aggregated chain.
"""

import torch
import numpy as np
import math
import matplotlib.pyplot as plt
from itertools import product

# ---- Device selection ----
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
# State space generation
# -----------------------
def get_product_state_space(d):
    """
    Generate the full state space for the Curie–Weiss model, i.e. all vectors in {-1,+1}^d.
    Returns a torch tensor of shape (2^d, d) with dtype=torch.int64.
    """
    grid = torch.tensor([-1, 1], dtype=torch.int64, device=device)
    grids = [grid for _ in range(d)]
    return torch.cartesian_prod(*grids)

# -----------------------
# Hamiltonian computation
# -----------------------
def compute_hamiltonian(x, h):
    """
    For a state x (tensor of shape (d,)), compute the Curie–Weiss Hamiltonian:
         H(x) = - (sum(x)^2) - h * sum(x)
    """
    s = torch.sum(x).float()
    return - (s * s) - h * s

# -----------------------
# Stationary distribution computation
# -----------------------
def stat_dist_MC_generation(state_space, beta, h):
    """
    Given the full state space (tensor of shape (M, d) of type int64),
    compute the Gibbs stationary distribution:
         pi(x) = exp(-beta * H(x)) / Z.
    Returns a tensor pi of shape (M,).
    """
    M = state_space.shape[0]
    energies = torch.empty(M, dtype=torch.float32, device=device)
    for i in range(M):
        energies[i] = compute_hamiltonian(state_space[i], h)
    unnorm = torch.exp(-beta * energies)
    Z = torch.sum(unnorm)
    pi = unnorm / Z
    return pi

# -----------------------
# Transition matrix computation (Glauber dynamics)
# -----------------------
def compute_transition_matrix(state_space, beta, h):
    """
    Compute the transition matrix P for the Glauber dynamics:
    For a state x, a coordinate is chosen uniformly at random.
    For a neighbor y (x with one coordinate flipped), the acceptance probability is:
          (1/d)*exp(-beta * (H(y)-H(x))_+),
    and the self-loop probability is chosen so that each row sums to 1.
    Returns the transition matrix P as a torch tensor of shape (M, M).
    """
    M, d = state_space.shape
    P = torch.zeros((M, M), dtype=torch.float32, device=device)
    
    pi_dummy = stat_dist_MC_generation(state_space, beta, h)
    energies = torch.empty(M, dtype=torch.float32, device=device)
    for i in range(M):
        energies[i] = compute_hamiltonian(state_space[i], h)
    
    state_to_index = {tuple(state_space[i].tolist()): i for i in range(M)}
    
    for i in range(M):
        x = state_space[i]
        total_rate = 0.0
        for j in range(d):
            y = x.clone()
            y[j] = -y[j]
            y_tuple = tuple(y.tolist())
            k = state_to_index[y_tuple]
            delta = energies[k] - energies[i]
            acc = math.exp(-beta * max(delta, 0))
            rate = (1.0 / d) * acc
            P[i, k] = rate
            total_rate += rate
        P[i, i] = 1.0 - total_rate
    return P

# -----------------------
# Aggregation and Entropy Rate
# -----------------------
def keep_S_in_mat(P, state_space, pi, S):
    """
    Aggregate the chain by projecting each state onto the coordinates in S.
    Two states are aggregated if they have the same spins on the coordinates in S.
    Returns:
       - partial_list: list of aggregated state tuples,
       - pi_S: aggregated stationary distribution (tensor),
       - P_S: aggregated transition matrix (tensor).
    """
    M, d = state_space.shape
    S_list = sorted(list(S))
    free_states = state_space.cpu().numpy()
    partial_map = {}
    partial_list = []
    full_to_reduced = np.empty(M, dtype=np.int32)
    for i, x in enumerate(free_states):
        xS = tuple(x[k] for k in S_list)
        if xS not in partial_map:
            partial_map[xS] = len(partial_map)
            partial_list.append(xS)
        full_to_reduced[i] = partial_map[xS]
    num_reduced = len(partial_list)
    pi_S_np = np.zeros(num_reduced, dtype=np.float32)
    P_cpu = P.cpu().numpy()
    pi_cpu = pi.cpu().numpy()
    for i in range(M):
        pi_S_np[full_to_reduced[i]] += pi_cpu[i]
    P_S_num = np.zeros((num_reduced, num_reduced), dtype=np.float32)
    for i in range(M):
        for j in range(M):
            P_S_num[full_to_reduced[i], full_to_reduced[j]] += pi_cpu[i] * P_cpu[i, j]
    P_S_np_final = np.zeros_like(P_S_num)
    for i in range(num_reduced):
        if pi_S_np[i] > 1e-15:
            P_S_np_final[i, :] = P_S_num[i, :] / pi_S_np[i]
    pi_S = torch.tensor(pi_S_np, dtype=torch.float32, device=device)
    P_S = torch.tensor(P_S_np_final, dtype=torch.float32, device=device)
    return partial_list, pi_S, P_S

def leave_S_out_mat(P, state_space, pi, S):
    Sbar = set(range(state_space.shape[1])) - set(S)
    return keep_S_in_mat(P, state_space, pi, Sbar)

def compute_entropy_rate(P, pi):
    """
    Compute the entropy rate of the chain:
         H(P,pi) = - sum_x pi(x) sum_y P(x,y) log(P(x,y)).
    """
    eps = 1e-15
    return -torch.sum(pi.unsqueeze(1) * P * torch.log(P + eps)).float()

def distorted_greedy_k_submod(g, c, V, m, k):
    """
    Generalized distorted greedy for k-submodular optimization:
    We want to approximately maximize f(S) = g(S) - c(S),
    where g is k-submodular & monotone, c is a nonnegative modular function,
    S = (S[0], ..., S[k-1]) with S[j] ⊆ V[j].
    V is a list/tuple: V[j] is the "universe" for the j-th coordinate.
    m is total cardinality bound: sum_j |S[j]| ≤ m.
    """
    S = [set() for _ in range(k)]

    for i in range(m):
        best_gain = float('-inf')
        best_j    = None
        best_e    = None

        factor = (1.0 - 1.0/m)**(m - (i+1))

        for j in range(k):
            for e in (V[j] - S[j]):
                old_g = g(S)
                old_c = c(S)
                S[j].add(e)
                new_g = g(S)
                new_c = c(S)
                S[j].remove(e)

                inc_g = new_g - old_g
                cost_e = new_c - old_c
                gain = factor*inc_g - cost_e

                if gain > best_gain:
                    best_gain = gain
                    best_j = j
                    best_e = e

        if best_gain > 0:
            S[best_j].add(best_e)

    return S

def plot_objective_per_iteration(f_values):
    plt.figure()
    plt.plot(range(1, len(f_values)+1), f_values, marker='o')
    plt.xlabel("Subset size", fontsize=16)
    plt.ylabel("Entropy rate", fontsize=16)
    plt.title("Entropy rate of the output against subset size", fontsize=14)
    plt.xticks(range(1, len(f_values)+1))
    plt.grid(True)
    plt.savefig("greedy_curie_weiss.pdf")
    plt.show()

# -----------------------
# MAIN
# -----------------------
if __name__ == "__main__":
    # Parameters for the Curie–Weiss model:
    d = 5            # number of spins
    beta = 0.1        # inverse temperature
    h = 0.0           # external magnetic field

    k = 3
    V = [
        set([0, 1]),
        set([2]),
        set([3, 4])
    ]

    state_space = get_product_state_space(d)
    P = compute_transition_matrix(state_space, beta, h)
    print(f"Generated state space with {state_space.shape[0]} states (dimension = {d}).")

    pi = stat_dist_MC_generation(state_space, beta, h)
    base_entropy = compute_entropy_rate(compute_transition_matrix(state_space, beta, h), pi).item()
    print(f"Entropy rate of full chain = {base_entropy}")

    def c(S):
        total_cost = 0.0
        for j, S_j in enumerate(S):
            for elem in S_j:
                Vj_minus_elem = set(V[j]) - {elem}
                
                _, pi_keep, P_keep = keep_S_in_mat(P, state_space, pi, Vj_minus_elem)
                _, pi_leave, P_leave = leave_S_out_mat(P, state_space, pi, V[j])
                
                val_keep = compute_entropy_rate(P_keep, pi_keep).item()
                val_leave = compute_entropy_rate(P_leave, pi_leave).item()
                
                total_cost += (val_keep - val_leave)
        return total_cost

    def f(S):
        total = 0.0
        for coords_j in S:
            if coords_j:
                _, piS, PS = keep_S_in_mat(P, state_space, pi, coords_j)
                total += compute_entropy_rate(PS, piS).item()
        return total

    def g(S):
        return f(S) + c(S)
    
    for m in range(1, d + 1):
        S = distorted_greedy_k_submod(g, c, V, m, k)
        print(f"Cardinality constraint {m}; Subset chosen: {S}; Value: {f(S)}")
