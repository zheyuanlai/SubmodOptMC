#!/usr/bin/env python

import torch
import numpy as np
import math
import matplotlib.pyplot as plt
from itertools import product

# -----------------------------------------------------------------------
#  Device Selection
# -----------------------------------------------------------------------
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS device")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA device")
else:
    device = torch.device("cpu")
    print("Using CPU device")


# -----------------------------------------------------------------------
#  Bernoulli–Laplace: Hahn polynomials, product state space, etc.
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
    Evaluate univariate Hahn polynomial:
    Q_n(x; alpha, beta, N_local) = sum_{j=0}^{n} [(-n)_j (-x)_j (n+alpha+beta+1)_j] /
                                   [ (alpha+1)_j (-N_local+1)_j * j! ].
    """
    s_val = 0.0
    eps = 1e-10
    for j in range(n+1):
        num = poch(-n, j) * poch(-x, j) * poch(n + alpha + beta + 1, j)
        den = poch(alpha+1, j) * (poch(-N_local+1, j) + eps) * math.factorial(j)
        s_val += num/den
    return s_val

def falling_factorial(a, k):
    """
    a_[k] = a*(a-1)*...*(a-k+1).
    """
    result = 1
    for i in range(k):
        result *= (a - i)
    return result

def compute_eigenvalues(N, s, l_values):
    """
    Compute eigenvalues β_n for the Bernoulli–Laplace level model:
      β_n = sum_{k=0}^{n} {n choose k} * (N-s)_{[n-k]}*(s)_{[k]} /
                            [(N)_{[n-k]}*(L-N)_{[k]}],
    where L = sum(l_values). Note that by design we have L > N.
    """
    l_total = sum(l_values)
    eigenvals = []
    for n in range(N+1):
        sum_term = 0.0
        for k in range(n+1):
            num = math.comb(n, k) * falling_factorial(N-s, n-k) * falling_factorial(s, k)
            den = falling_factorial(N, n-k) * falling_factorial(l_total - N, k)
            sum_term += num/den
        eigenvals.append(sum_term)
    return eigenvals

def get_product_state_space(num_free):
    """
    Construct the product-form state space corresponding to the free coordinates.
    Here each free coordinate is binary (0 or 1).
    Returns a list of tuples of length num_free.
    """
    # The product state space is simply {0,1}^(num_free)
    return list(product([0,1], repeat=num_free))

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
    Compute stationary distribution π( x ) = [∏_{i=1}^{d} (l_i choose x_i)] / ( (∑ l_i choose N) )
    For product state space, each state x is a tuple of length (d-1) corresponding to the free coordinates.
    The full state is then defined as:
      x_full = (x[0], ..., x[d-2], x_{d}) where x_{d} = N - sum(x).
    """
    d = len(l_values)
    l_total = sum(l_values)
    denom = math.comb(l_total, N)
    pi_vals = []
    for x in state_space:
        # form the full state: first (d-1) coordinates from x, last coordinate is fixed.
        x_full = list(x) + [N - sum(x)]
        num = 1
        for xi, li in zip(x_full, l_values):
            num *= math.comb(li, xi)
        pi_vals.append(num/denom)
    arr = np.array(pi_vals, dtype=np.float32)
    return arr / arr.sum()

def get_multi_indices(total, length):
    """
    Generate all tuples (m1,.., m_length) of nonnegative integers summing to 'total'.
    """
    if length == 1:
        return [(total,)]
    result = []
    def recurse(tleft, dim, partial):
        if dim == 1:
            result.append(tuple(partial + [tleft]))
            return
        for v in range(tleft+1):
            recurse(tleft - v, dim - 1, partial + [v])
    recurse(total, length, [])
    return result

def multivariate_hahn(m, x, l_values, N):
    """
    Evaluate the multivariate Hahn polynomial:
       Q_m( x ; N, -∑_{i=1}^{d} l_i)
    where x should be the full state vector.
    For product-form state space, if x is of length d-1, we first append
    x_d = N - sum(x).
    We then use:
       Q_m( x ) = ∏_{i=1}^{d-1} Q_{m_i}( x_i; alpha_i, beta_i, N_i ),
    with alpha_i = l_i - 1, beta_i = (sum_{j=i+1}^{d} l_j) - 1, and N_i defined appropriately.
    """
    d = len(l_values)
    # If x has length d-1, append x_d.
    if len(x) == d - 1:
        x = list(x) + [N - sum(x)]
    prod = 1.0
    s_val = 0  # used to update N_i below
    for i in range(d-1):
        alpha_i = l_values[i] - 1
        beta_i = sum(l_values[i+1:]) - 1
        # For our model, we use N_i = N - s_val.
        N_i = N - s_val
        xi = x[i]
        prod *= univariate_hahn(m[i], xi, alpha_i, beta_i, N_i)
        s_val += xi
    return prod

# -----------------------------------------------------------------------
#  Markov Chain Generation in Torch
# -----------------------------------------------------------------------
def torch_MC_generation(N, d, l_values, s, product_form=True):
    """
    Return (state_space, pi, P) for the Bernoulli–Laplace chain, in float32.
    If product_form=True, then we assume that the first d-1 coordinates are free,
    taking values in {0,1}, and the d-th coordinate is given by x_d = N - sum(x).
    Also, parameters are chosen such that sum(l_values) > N.
    """
    if product_form:
        num_free = d - 1  # free coordinates
        state_space = get_product_state_space(num_free)
    else:
        state_space = get_state_space(N, d)  # original partition state space (not product)
    num_states = len(state_space)
    pi_np = stat_dist_MC_generation(state_space, l_values, N)
    # Use float32 for pi
    pi = torch.tensor(pi_np, dtype=torch.float32, device=device)

    eigenvals = compute_eigenvalues(N, s, l_values)

    M_n_list = []
    for n in range(N+1):
        multi_indices_n = get_multi_indices(n, d-1)  # using d-1 free indices for eigenfunctions
        M_n = np.zeros((num_states, len(multi_indices_n)), dtype=np.float32)
        for i, x in enumerate(state_space):
            for j, m in enumerate(multi_indices_n):
                val = multivariate_hahn(m, x, l_values, N)
                M_n[i, j] = val
        M_n_list.append(torch.tensor(M_n, dtype=torch.float32, device=device))

    A = torch.zeros((num_states, num_states), dtype=torch.float32, device=device)
    for n in range(N+1):
        beta_n = eigenvals[n]
        M_n = M_n_list[n]
        A_n = torch.matmul(M_n, M_n.t())
        A += beta_n * A_n

    A = torch.clamp(A, min=0.0)
    P = A * pi.unsqueeze(0)
    row_sums = P.sum(dim=1, keepdim=True)
    P = P / (row_sums + 1e-15)
    P = torch.clamp(P, min=0.0)
    row_sums2 = P.sum(dim=1, keepdim=True)
    P = P / (row_sums2 + 1e-15)
    
    return state_space, pi.float(), P.float()

# -----------------------------------------------------------------------
#  Aggregation: keep / leave subset of coordinates
# -----------------------------------------------------------------------
def keep_S_in_mat(P, state_space, pi, S):
    """
    Aggregate the chain (P, pi) onto coordinates S.
    Return (partial_states, pi_S, P_S) where:
      - partial_states: list of states restricted to S,
      - pi_S[i] = sum_{x: x|S = partial_states[i]} pi(x),
      - P_S[i,j] = (1/pi_S[i]) * sum_{x|S=i, y|S=j} pi(x) P(x,y).
    """
    M = len(state_space)
    S_list = sorted(list(S))
    pi_cpu = pi.detach().cpu().numpy()
    P_cpu = P.detach().cpu().numpy()
    partial_map = {}
    partial_list = []
    next_idx = 0
    full_to_reduced = np.empty(M, dtype=np.int32)
    for i, x in enumerate(state_space):
        # Here, x is a tuple representing the free coordinates.
        xS = tuple(x[k] for k in S_list)
        if xS not in partial_map:
            partial_map[xS] = next_idx
            partial_list.append(xS)
            next_idx += 1
        full_to_reduced[i] = partial_map[xS]
    num_reduced = len(partial_list)
    pi_S_np = np.zeros(num_reduced, dtype=np.float32)
    for i in range(M):
        pi_S_np[full_to_reduced[i]] += pi_cpu[i]
    P_S_num = np.zeros((num_reduced, num_reduced), dtype=np.float32)
    for x_idx in range(M):
        xS_idx = full_to_reduced[x_idx]
        w_x = pi_cpu[x_idx]
        if w_x < 1e-15:
            continue
        for y_idx, p_xy in enumerate(P_cpu[x_idx]):
            if p_xy < 1e-15:
                continue
            yS_idx = full_to_reduced[y_idx]
            P_S_num[xS_idx, yS_idx] += w_x * p_xy
    P_S_np_final = np.zeros_like(P_S_num)
    for i in range(num_reduced):
        if pi_S_np[i] > 1e-15:
            P_S_np_final[i, :] = P_S_num[i, :] / pi_S_np[i]
    pi_S = torch.tensor(pi_S_np, dtype=torch.float32, device=device)
    P_S = torch.tensor(P_S_np_final, dtype=torch.float32, device=device)
    return partial_list, pi_S, P_S

def leave_S_out_mat(P, state_space, pi, S):
    """
    Return the aggregation matrix for the complement subset of S.
    """
    d = len(state_space[0])  # free dimensions
    Sbar = set(range(d)) - set(S)
    return keep_S_in_mat(P, state_space, pi, Sbar)

# -----------------------------------------------------------------------
#  Entropy Rate
# -----------------------------------------------------------------------
def compute_entropy_rate(P, pi):
    """
    Compute entropy rate: H(P) = - sum_{x,y} pi[x] P[x,y] log(P[x,y]).
    """
    eps = 1e-15
    pi_ = pi.float()
    P_ = P.float()
    return -torch.sum(pi_.unsqueeze(1) * P_ * torch.log(P_ + eps)).float()

# -----------------------------------------------------------------------
#  Greedy
# -----------------------------------------------------------------------
def greedy(f, X, k):
    """
    Greedy algorithm for submodular maximization.
    """
    S = set()
    plot_vals = []
    for i in range(k):
        gains = [(f(S.union({e})) - f(S), e) for e in X - S]
        gain, elem = max(gains)
        if gain >= 0:
            S.add(elem)
        print(f"Iteration {i+1}, S = {S}, Value = {f(S)}")
        plot_vals.append(f(S))
    return plot_vals

def plot_objective_per_iteration(f_values):
    """
    Plot objective values per iteration.
    """
    iters = range(1, len(f_values)+1)
    plt.figure(figsize=(6,4))
    plt.plot(iters, f_values, marker='o')
    plt.title("Entropy rate vs. subset size")
    plt.xlabel("Subset size")
    plt.ylabel("Entropy rate")
    plt.grid(True)
    plt.show()

# -----------------------------------------------------------------------
#  MAIN
# -----------------------------------------------------------------------
if __name__=="__main__":
    # Experimental parameters for product state space:
    N = 10
    d = 11  # total number of types; free coordinates = d-1 = 15, yielding state space {0,1}^{15}
    # Choose l1 = ... = l15 = 1 and l16 = 15 so that sum(l_values)=15+15=30 > N=15.
    l_values = [1]*10 + [15]
    s = 1

    # Generate chain with product state space:
    state_space, pi, P = torch_MC_generation(N, d, l_values, s, product_form=True)
    print(f"Generated Bernoulli–Laplace chain with product state space of dimension {d-1} (total states = {len(state_space)})")
    
    base_entropy = compute_entropy_rate(P, pi).item()
    print(f"Entropy rate of full chain = {base_entropy}")

    def submod_func(S):
        # S here is a subset of free coordinate indices {0,..., d-2}
        _, pi_S, P_S = keep_S_in_mat(P, state_space, pi, S)
        return compute_entropy_rate(P_S, pi_S).item()

    # Use free coordinates (0-indexed): set U = {0,1,..., d-2}
    U = set(range(d-1))
    # Run greedy submodular maximization over the free coordinates:
    f_values = greedy(submod_func, U, d-1)
    print("Objective values:", f_values)
    plot_objective_per_iteration(f_values)
