#!/usr/bin/env python

import torch
import numpy as np
import math
import matplotlib.pyplot as plt

# ------------------------------------------------------
#  Device selection
# ------------------------------------------------------
#if torch.backends.mps.is_available():
#    device = torch.device("mps")
#    print("Using MPS device")
#elif torch.cuda.is_available():
#    device = torch.device("cuda")
#    print("Using CUDA device")
#else:
#    device = torch.device("cpu")
#    print("Using CPU device")

device = 'cpu'

def poch(a, j):
    """Pochhammer symbol (a)_j = a*(a+1)*...*(a+j-1)."""
    result = 1.0
    for i in range(j):
        result *= (a + i)
    return result

def univariate_hahn(n, x, alpha, beta, N_local):
    """
    Q_n(x; alpha, beta, N_local) = sum_{j=0..n} [(-n)_j * (-x)_j * (n+alpha+beta+1)_j] /
                                                 [ (alpha+1)_j * (-N_local+1)_j * j! ].
    """
    s_val = 0.0
    eps = 1e-10
    for j in range(n+1):
        num = poch(-n, j)*poch(-x, j)*poch(n+alpha+beta+1, j)
        den = (poch(alpha+1, j)*poch(-N_local+1, j)*math.factorial(j) + eps)
        s_val += num/den
    return s_val

def falling_factorial(a, k):
    """a_[k] = a * (a-1) * ... * (a-k+1)."""
    result = 1
    for i in range(k):
        result *= (a - i)
    return result

def compute_eigenvalues(N, s, l_values):
    """
    Bernoulli–Laplace eigenvalues. 
    Beta_n = sum_{k=0..n} [ (nCk)*(N-s)_[n-k]*s_[k] ] / [ N_[n-k]*(L - N)_[k] ],
    where L = sum(l_values).
    """
    l_total = sum(l_values)
    eigenvals = []
    for n in range(N+1):
        sum_term = 0.0
        for k in range(n+1):
            num = (math.comb(n,k)
                   * falling_factorial(N - s, n - k)
                   * falling_factorial(s, k))
            den = (falling_factorial(N, n - k)
                   * falling_factorial(l_total - N, k))
            sum_term += num/den
        eigenvals.append(sum_term)
    return eigenvals

def get_state_space(N, d):
    """
    All d-tuples x in N^d with sum(x)=N.
    Returns a list of tuples, length = comb(N + d - 1, d - 1).
    """
    if d == 1:
        return [(N,)]
    result = []
    def recurse(remaining, dim, partial):
        if dim == 1:
            result.append(tuple(partial+[remaining]))
            return
        for v in range(remaining+1):
            recurse(remaining-v, dim-1, partial+[v])
    recurse(N, d, [])
    return result

def stat_dist_MC_generation(state_space, l_values, N):
    """
    pi(x) = [ ∏_i comb(l_i, x_i) ] / comb( sum(l_values), N ).
    """
    l_sum = sum(l_values)
    denom = math.comb(l_sum, N)
    pi_vals = []
    for x in state_space:
        num = 1
        for xi, li in zip(x, l_values):
            num *= math.comb(li, xi)
        pi_vals.append(num/denom)
    arr = np.array(pi_vals, dtype=np.float64)
    return arr / arr.sum()

def get_multi_indices(total, length):
    """Generate all tuples (m1,..,m_length) of nonnegative ints summing to 'total'."""
    if length == 1:
        return [(total,)]
    result = []
    def recurse(tleft, dim, partial):
        if dim==1:
            result.append(tuple(partial+[tleft]))
            return
        for v in range(tleft+1):
            recurse(tleft-v, dim-1, partial+[v])
    recurse(total, length, [])
    return result

def multivariate_hahn(m, x, l_values, N):
    """
    Q_m(x) = ∏_{i=1}^{d-1} Q_{m_i}( x_i; alpha_i, beta_i, N_i ).
    """
    d = len(l_values)
    prod = 1.0
    s_val=0
    for i in range(d-1):
        alpha_i = l_values[i] - 1
        beta_i  = sum(l_values[i+1:]) - 1
        N_i     = N - s_val
        x_i     = x[i]
        prod   *= univariate_hahn(m[i], x_i, alpha_i, beta_i, N_i)
        s_val  += x_i
    return prod

def torch_MC_generation(N, d, l_values, s):
    """
    Return (state_space, pi, P) for the Bernoulli–Laplace chain, in double precision.
    """
    state_space = get_state_space(N, d)
    num_states  = len(state_space)
    pi_np       = stat_dist_MC_generation(state_space, l_values, N)
    pi          = torch.tensor(pi_np, dtype=torch.float64, device=device)

    eigenvals   = compute_eigenvalues(N, s, l_values)

    # Build M_n
    M_n_list = []
    for n in range(N+1):
        multi_indices_n = get_multi_indices(n, d-1)
        M_n = np.zeros((num_states, len(multi_indices_n)), dtype=np.float64)
        for i, x in enumerate(state_space):
            for j, m in enumerate(multi_indices_n):
                val = multivariate_hahn(m, x, l_values, N)
                M_n[i,j] = val
        M_n_list.append(torch.tensor(M_n, dtype=torch.float64, device=device))

    A = torch.zeros((num_states, num_states), dtype=torch.float64, device=device)
    for n in range(N+1):
        beta_n = eigenvals[n]
        M_n    = M_n_list[n]
        A_n    = torch.matmul(M_n, M_n.t())  # double
        A     += beta_n*A_n

    A = torch.clamp(A, min=0.0)

    # Build P
    P = A * pi.unsqueeze(0)
    row_sums = P.sum(dim=1, keepdim=True)
    P = P / (row_sums + 1e-15)
    P = torch.clamp(P, min=0.0)
    row_sums2 = P.sum(dim=1, keepdim=True)
    P = P / (row_sums2 + 1e-15)

    pi_32 = pi.float()
    P_32  = P.float()
    return state_space, pi_32, P_32

def keep_S_in_mat(P, state_space, pi, S):
    """
    Aggregates chain (P, pi) onto the coordinates in S (subset of {0..d-1}).
    Return partial_list, pi_S, P_S
    """
    M = len(state_space)
    S_list = sorted(list(S))

    pi_cpu = pi.detach().cpu().numpy()
    P_cpu  = P.detach().cpu().numpy()

    partial_map = {}
    partial_list = []
    next_idx     = 0
    full_to_reduced = np.empty(M, dtype=np.int64)

    for i, x in enumerate(state_space):
        xS = tuple(x[k] for k in S_list)
        if xS not in partial_map:
            partial_map[xS] = next_idx
            partial_list.append(xS)
            next_idx += 1
        full_to_reduced[i] = partial_map[xS]

    num_reduced = len(partial_list)

    pi_S_np = np.zeros(num_reduced, dtype=np.float64)
    for i in range(M):
        pi_S_np[ full_to_reduced[i] ] += pi_cpu[i]

    P_S_num = np.zeros((num_reduced, num_reduced), dtype=np.float64)
    for x_idx in range(M):
        xS_idx = full_to_reduced[x_idx]
        w_x    = pi_cpu[x_idx]
        if w_x<1e-15:
            continue
        row_x  = P_cpu[x_idx]
        for y_idx, p_xy in enumerate(row_x):
            if p_xy<1e-15:
                continue
            yS_idx = full_to_reduced[y_idx]
            P_S_num[xS_idx, yS_idx] += w_x*p_xy

    P_S_np_final = np.zeros_like(P_S_num)
    for i in range(num_reduced):
        if pi_S_np[i]>1e-15:
            P_S_np_final[i, :] = P_S_num[i, :] / pi_S_np[i]

    pi_S = torch.tensor(pi_S_np, dtype=torch.float32, device=device)
    P_S  = torch.tensor(P_S_np_final, dtype=torch.float32, device=device)
    return partial_list, pi_S, P_S

def leave_S_out_mat(P, state_space, pi, S):
    """Keep the complement of S in the chain."""
    d = len(state_space[0])
    Sbar = set(range(d)) - set(S)
    return keep_S_in_mat(P, state_space, pi, Sbar)

def compute_entropy_rate(P, pi):
    """
    H(P) = - sum_{x,y} pi[x]*P[x,y]*log(P[x,y]).
    """
    pi_64 = pi.double()
    P_64  = P.double()
    val = -torch.sum(pi_64.unsqueeze(1)* P_64 * torch.log(P_64 + 1e-15))
    return val.float()

# ------------------------------------------------------
#  Distorted Greedy for k-submodular (Algorithm 3)
# ------------------------------------------------------
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
    f_values = []

    for i in range(m):
        best_gain = float('-inf')
        best_j    = None
        best_e    = None

        current_size = sum(len(S_j) for S_j in S)
        if current_size >= m:
            break

        for j in range(k):
            for e in (V[j] - S[j]):
                if current_size + 1 > m:
                    continue

                factor = (1.0 - 1.0/m)**(m - (i+1))

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

        if best_gain > 0 and (best_j is not None) and (best_e is not None):
            S[best_j].add(best_e)

        print(f"Step {i+1}: Set chosen: {[[(elem + 1) for elem in S_j] for S_j in S]}, f(S) = {g(S) - c(S)}")
        f_values.append(g(S) - c(S))

    return S, f_values

# ------------------------------------------------------
#  MAIN Example
# ------------------------------------------------------
if __name__ == "__main__":
    N = 3
    d = 15
    l_values = [5]*d
    s = 1

    state_space, pi, P = torch_MC_generation(N, d, l_values, s)
    print(f"Chain dimension d={d}, #states = {len(state_space)}.")

    base_entropy = compute_entropy_rate(P, pi).item()
    print("Full chain's entropy rate:", base_entropy)

    k = 3

    V = [
        set(range(0,6)),
        set(range(6,12)),
        set(range(12,15))
    ]

    def k_modular_c(S):
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

    def k_submod_g(S):
        total = 0.0
        for coords_j in S:
            if coords_j:
                _, piS, PS = keep_S_in_mat(P, state_space, pi, coords_j)
                total += compute_entropy_rate(PS, piS).item()

        total += k_modular_c(S)
        return total

    m = 15

    chosen_S, f_vals = distorted_greedy_k_submod(k_submod_g, k_modular_c, V, m, k)

    print("\nDistorted Greedy completed.")
    print("Objective values by iteration:", f_vals)

    plt.figure()
    plt.plot(range(1, len(f_vals)+1), f_vals, marker='o')
    plt.xlabel("Subset size")
    plt.ylabel("Summed entropy rates")
    plt.title("Summed entropy rates of output of generalized distorted greedy against subset size")
    plt.grid(True)
    plt.show()