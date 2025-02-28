#!/usr/bin/env python

import torch
import numpy as np
import math
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------
#  Device Selection
# -----------------------------------------------------------------------
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
# -----------------------------------------------------------------------
#  Bernoulli–Laplace: Hahn polynomials, state space, etc.
# -----------------------------------------------------------------------

def poch(a, j):
    """
    Pochhammer symbol (a)_j = a (a+1) ... (a+j-1) with (a)_0=1.
    """
    result = 1.0
    for i in range(j):
        result *= (a + i)
    return result

def univariate_hahn(n, x, alpha, beta, N_local):
    """
    Q_n(x; alpha, beta, N_local) = sum_{j=0..n} [(-n)_j * (-x)_j * (n+alpha+beta+1)_j]
                                         / [ (alpha+1)_j * (-N_local+1)_j * j! ].
    """
    s_val = 0.0
    eps = 1e-10
    for j in range(n+1):
        num = poch(-n, j)*poch(-x, j)*poch(n+alpha+beta+1, j)
        den = (poch(alpha+1, j)*poch(-N_local+1, j)*math.factorial(j) + eps)
        s_val += num/den
    return s_val

def falling_factorial(a, k):
    """
    a_[k] = a * (a-1) * ... * (a-k+1).
    """
    result = 1
    for i in range(k):
        result *= (a - i)
    return result

def compute_eigenvalues(N, s, l_values):
    """
    Bernoulli–Laplace eigenvalues β_n, for n=0..N, with:
      β_n = sum_{k=0..n} [ (n choose k) * (N-s)_[n-k] * (s)_[k ] ] /
                            [ (N)_[n-k] * (L - N)_[k] ],
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
    if d==1:
        return [(N,)]
    result = []
    def recurse(remaining, dim, partial):
        if dim==1:
            result.append(tuple(partial+[remaining]))
            return
        for v in range(remaining+1):
            recurse(remaining-v, dim-1, partial+[v])
    recurse(N, d, [])
    return result

def stat_dist_MC_generation(state_space, l_values, N):
    """
    pi(x) = [ ∏_i comb(l_i, x_i ) ] / comb( sum(l_values), N ).
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
    """
    Generate all tuples (m1,..,m_length) of nonnegative ints summing to 'total'.
    """
    if length==1:
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
        alpha_i = l_values[i]-1
        beta_i  = sum(l_values[i+1:]) -1
        N_i     = N - s_val
        x_i     = x[i]
        prod   *= univariate_hahn(m[i], x_i, alpha_i, beta_i, N_i)
        s_val  += x_i
    return prod

# -----------------------------------------------------------------------
#  Markov Chain Generation in Torch
# -----------------------------------------------------------------------
def torch_MC_generation(N, d, l_values, s):
    """
    Return (state_space, pi, P) for the Bernoulli–Laplace chain, in double precision,
    with partial checks/clamping to reduce overflow.
    """
    # 1) Build state space & pi
    state_space = get_state_space(N, d)
    num_states  = len(state_space)
    pi_np       = stat_dist_MC_generation(state_space, l_values, N)
    # Use double precision for pi
    pi          = torch.tensor(pi_np, dtype=torch.float64, device=device)

    # 2) Eigenvalues
    eigenvals   = compute_eigenvalues(N, s, l_values)

    # 3) Build M_n in double precision, possibly with manual scaling
    M_n_list = []
    for n in range(N+1):
        multi_indices_n = get_multi_indices(n, d-1)
        # float64
        M_n = np.zeros((num_states, len(multi_indices_n)), dtype=np.float64)
        for i, x in enumerate(state_space):
            for j, m in enumerate(multi_indices_n):
                val = multivariate_hahn(m, x, l_values, N)
                M_n[i,j] = val
        # Convert to torch double
        M_n_list.append(torch.tensor(M_n, dtype=torch.float64, device=device))

    # 4) Build A = sum_{n} beta_n * (M_n @ M_n.T)
    A = torch.zeros((num_states, num_states), dtype=torch.float64, device=device)
    for n in range(N+1):
        beta_n = eigenvals[n]
        M_n    = M_n_list[n]
        A_n    = torch.matmul(M_n, M_n.t())  # double
        A     += beta_n*A_n

    # Clamp negative
    A = torch.clamp(A, min=0.0)

    # 5) Build P(x,y) = pi[y] * A(x,y); row normalize
    # still double
    P = A * pi.unsqueeze(0)
    row_sums = P.sum(dim=1, keepdim=True)
    P = P / (row_sums + 1e-15)

    # clamp again
    P = torch.clamp(P, min=0.0)
    row_sums2 = P.sum(dim=1, keepdim=True)
    P = P / (row_sums2 + 1e-15)

    # If you prefer returning float32 at the end:
    pi_32 = pi.float()
    P_32  = P.float()
    return state_space, pi_32, P_32

# -----------------------------------------------------------------------
#  Aggregation: keep / leave subset of coordinates
# -----------------------------------------------------------------------

def keep_S_in_mat(P, state_space, pi, S):
    """
    Aggregates the Markov chain (P, pi) onto the coordinates in S.

    Parameters:
        P (torch.Tensor): The transition matrix of the Markov chain.
        state_space (list of tuples): The original state space.
        pi (torch.Tensor): The stationary distribution.
        S (set): The subset of indices to retain.
    
    Returns:
        tuple: (partial_states, pi_S, P_S) where
          - `partial_states` is a list of reduced states x_S,
          - `pi_S[i] = sum_{x: x|S = partial_states[i]} pi(x)`,
          - `P_S[i,j] = (1/pi_S[i]) * sum_{x|S=i} sum_{y|S=j} pi(x)*P(x,y)`.
    """
    M = len(state_space)
    S_list = sorted(list(S))  # Convert set to ordered list for indexing

    # Create mapping from full state to reduced state index
    partial_map = {}
    partial_list = []
    full_to_reduced = torch.empty(M, dtype=torch.int64, device=device)

    next_idx = 0
    for i, x in enumerate(state_space):
        x_S = tuple(x[k] for k in S_list)
        if x_S not in partial_map:
            partial_map[x_S] = next_idx
            partial_list.append(x_S)
            next_idx += 1
        full_to_reduced[i] = partial_map[x_S]

    num_reduced = len(partial_list)

    # Compute marginal pi_S
    pi_S = torch.zeros(num_reduced, dtype=torch.float64, device=device)
    for i in range(M):
        pi_S[full_to_reduced[i]] += pi[i]

    # Compute reduced transition matrix P_S
    P_S_num = torch.zeros((num_reduced, num_reduced), dtype=torch.float64, device=device)
    for x_idx in range(M):
        xS_idx = full_to_reduced[x_idx]
        w_x = pi[x_idx]
        if w_x < 1e-15:
            continue
        for y_idx in range(M):
            if P[x_idx, y_idx] < 1e-15:
                continue
            yS_idx = full_to_reduced[y_idx]
            P_S_num[xS_idx, yS_idx] += w_x * P[x_idx, y_idx]

    # Normalize rows by pi_S
    P_S = torch.zeros_like(P_S_num)
    for i in range(num_reduced):
        if pi_S[i] > 1e-15:
            P_S[i, :] = P_S_num[i, :] / pi_S[i]

    return partial_list, pi_S.to(torch.float32), P_S.to(torch.float32)

def leave_S_out_mat(P, state_space, pi, S):
    d = len(state_space[0])
    Sbar = set(range(d)) - set(S)
    return keep_S_in_mat(P, state_space, pi, Sbar)

# -----------------------------------------------------------------------
#  Entropy Rate
# -----------------------------------------------------------------------
def compute_entropy_rate(P, pi):
    """
    H(P) = - sum_{x,y} pi[x]*P[x,y]*log(P[x,y]).
    """
    pi_64 = pi.double()
    P_64  = P.double()
    val = -torch.sum(pi_64.unsqueeze(1)* P_64 * torch.log(P_64 + 1e-15))
    return val.float()

def KL_divergence_gpu(pi, P, Q):
    """
    Computes the Kullback-Leibler divergence KL(P||Q) on the GPU.
    P and Q are torch tensors (transition matrices).
    KL(P||Q) = sum_x pi[x] * sum_y P[x,y] * log(P[x,y]/Q[x,y]),
    where pi is the stationary distribution of P.
    """
    kl = torch.sum(pi.unsqueeze(1) * P * torch.log((P + 1e-10) / (Q + 1e-10)))
    return kl

def compute_outer_product_gpu(A, B):
    """
    Computes the outer (Kronecker) product of matrices A and B on the GPU.
    Uses torch.kron for efficiency.
    """
    return torch.kron(A, B)

# -----------------------------------------------------------------------
#  Distorted Greedy
# -----------------------------------------------------------------------
def distorted_greedy(f, c, U, m):
    """
    Distorted greedy with the set function f, cost c, ground set U.
    """
    S = set()
    for i in range(m):
        best_gain = float('-inf')
        best_e    = None
        for e in (U - S):
            gain = ((1 - 1/m)**(m - (i+1))) * (f(S | {e}) - f(S)) - c({e})
            if gain>best_gain:
                best_gain = gain
                best_e    = e
        if best_e is not None:
            check_gain = ((1 - 1/m)**(m - (i+1))) * (f(S|{best_e})- f(S)) - c({best_e})
            if check_gain>0:
                S.add(best_e)

    return S

def plot_objective_per_iteration(f_values):
    """
    f_values: list of submod function values over iterations.
    """
    iters = range(1, len(f_values)+1)
    plt.figure(figsize=(6,4))
    plt.plot(iters, f_values, marker='o')
    plt.title("Entropy rate of output of distorted greedy against subset size")
    plt.ylabel("Entropy rate")
    plt.xlabel("Subset size")
    plt.grid(True)
    plt.show()

# -----------------------------------------------------------------------
#  MAIN
# -----------------------------------------------------------------------
if __name__=="__main__":
    N = 3
    d = 15
    l_values = [5]*d
    s = 1

    state_space, pi, P = torch_MC_generation(N, d, l_values, s)
    print(f"Generated Bernoulli–Laplace chain with dimension d={d}. #states={len(state_space)}")

    def f(S):
        _, piS, PS = keep_S_in_mat(P, state_space, pi, S)
        _, piSbar, PSbar = leave_S_out_mat(P, state_space, pi, S)
        return KL_divergence_gpu(pi, P, compute_outer_product_gpu(PS, PSbar))
    
    def c(S):
        val = 0.0
        for e in S:
            _, pi_minus, P_minus = leave_S_out_mat(P, state_space, pi, {e})
            _, pi_e, P_e = keep_S_in_mat(P, state_space, pi, {e})
            val += KL_divergence_gpu(pi, P, compute_outer_product_gpu(P_minus, P_e))
        return val

    def g(S):
        return f(S) + c(S)

    U = set(range(d))
    f_values = []

    for m in range(1, d + 1):
        chosen_subset = distorted_greedy(g, c, U, m)
        f_val = f(chosen_subset)
        f_values.append(f_val)
        print(f"Cardinality constraint {m}; Subset chosen: {chosen_subset}; Value: {f_val}")

    print(f"\nDistorted Greedy finished. Subset chosen = {chosen_subset}")
    print("f-values:", f_values)

    plot_objective_per_iteration(f_values)