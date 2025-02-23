#!/usr/bin/env python

import torch
import numpy as np
import math
import matplotlib.pyplot as plt

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

#device = 'cpu'
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
    arr = np.array(pi_vals, dtype=np.float32)
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
    pi          = torch.tensor(pi_np, dtype=torch.float32, device=device)

    # 2) Eigenvalues
    eigenvals   = compute_eigenvalues(N, s, l_values)

    # 3) Build M_n in double precision, possibly with manual scaling
    M_n_list = []
    for n in range(N+1):
        multi_indices_n = get_multi_indices(n, d-1)
        # float64
        M_n = np.zeros((num_states, len(multi_indices_n)), dtype=np.float32)
        for i, x in enumerate(state_space):
            for j, m in enumerate(multi_indices_n):
                val = multivariate_hahn(m, x, l_values, N)
                M_n[i,j] = val
        # Convert to torch double
        M_n_list.append(torch.tensor(M_n, dtype=torch.float32, device=device))

    # 4) Build A = sum_{n} beta_n * (M_n @ M_n.T)
    A = torch.zeros((num_states, num_states), dtype=torch.float32, device=device)
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
    Aggregates the full chain (P, pi) onto coordinates S.
    
    Return (partial_states, pi_S, P_S):
      - partial_states is a list of new states x_S,
      - pi_S[i] = sum_{x: x|S = partial_states[i]} pi(x),
      - P_S[i,j] = (1/pi_S[i]) * sum_{x|S=i} sum_{y|S=j} pi(x)*P(x,y).
    """
    M = len(state_space)
    S_list = sorted(list(S))

    # Convert to CPU for loops
    pi_cpu = pi.detach().cpu().numpy()
    P_cpu  = P.detach().cpu().numpy()

    # Map each full state -> partial state index
    partial_map = {}
    partial_list = []
    next_idx     = 0
    full_to_reduced = np.empty(M, dtype=np.int32)

    for i, x in enumerate(state_space):
        xS = tuple(x[k] for k in S_list)
        if xS not in partial_map:
            partial_map[xS] = next_idx
            partial_list.append(xS)
            next_idx += 1
        full_to_reduced[i] = partial_map[xS]

    num_reduced = len(partial_list)

    # Accumulate pi_S
    pi_S_np = np.zeros(num_reduced, dtype=np.float32)
    for i in range(M):
        pi_S_np[ full_to_reduced[i] ] += pi_cpu[i]

    # Accumulate transitions
    P_S_num = np.zeros((num_reduced, num_reduced), dtype=np.float32)
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

    # Divide each row by pi_S
    P_S_np_final = np.zeros_like(P_S_num)
    for i in range(num_reduced):
        if pi_S_np[i]>1e-15:
            P_S_np_final[i, :] = P_S_num[i, :] / pi_S_np[i]
        # else row stays 0

    # Convert back to torch
    pi_S = torch.tensor(pi_S_np, dtype=torch.float32, device=device)
    P_S  = torch.tensor(P_S_np_final, dtype=torch.float32, device=device)

    return partial_list, pi_S, P_S

def leave_S_out_mat(P, state_space, pi, S):
    """
    Complement subset: we keep all coords except those in S.
    """
    d = len(state_space[0])  # original dimension
    Sbar = set(range(d)) - set(S)
    return keep_S_in_mat(P, state_space, pi, Sbar)

# -----------------------------------------------------------------------
#  Entropy Rate
# -----------------------------------------------------------------------
def compute_entropy_rate(P, pi):
    """
    H(P) = - sum_{x,y} pi[x]*P[x,y]*log(P[x,y]).
    """
    pi_64 = pi.float()
    P_64  = P.float()
    # Because we might have P=0 => log(0) = -inf => 0 * -inf => nan,
    # we do a small epsilon inside log.
    val = -torch.sum( pi_64.unsqueeze(1)* P_64 * torch.log(P_64 + 1e-15) )
    return val.float()

# -----------------------------------------------------------------------
#  Greedy
# -----------------------------------------------------------------------
def greedy(f, X, k):
    """
    Greedy algorithm for submodular maximization with cardinality constraints.
    """
    S = set()
    plot_vals = []
    for i in range(k):
        gains = [(f(S.union({e})) - f(S), e) for e in X - S]
        gain, elem = max(gains)
        if gain >= 0: S.add(elem)
        print(f"Iteration {i+1}, S = {S}")
        plot_vals.append(f(S))
    return plot_vals

def plot_objective_per_iteration(f_values):
    """
    f_values: list of submod function values over iterations.
    """
    iters = range(1, len(f_values)+1)
    plt.figure(figsize=(6,4))
    plt.plot(iters, f_values, marker='o')
    plt.title("Entropy rate of the output of greedy algorithm against subset size")
    plt.xlabel("Subset size")
    plt.ylabel("Entropy rate")
    plt.grid(True)
    plt.show()

# -----------------------------------------------------------------------
#  MAIN
# -----------------------------------------------------------------------
if __name__=="__main__":
    # For demonstration:
    N = 8
    d = 15
    l_values = [5]*d
    s = 1

    # Generate the chain with debug checks
    state_space, pi, P = torch_MC_generation(N, d, l_values, s)
    print(f"Generated Bernoulli–Laplace chain with dimension d={d}. #states={len(state_space)}")

    # We'll compute the base chain's entropy once
    base_entropy = compute_entropy_rate(P, pi).item()
    print(f"Entropy rate of the full chain = {base_entropy}")

    def submod_func(S):
        _, piS, PS = keep_S_in_mat(P, state_space, pi, S)
        return compute_entropy_rate(PS, piS).item()

    # Distorted Greedy
    U = set(range(d))
    f_values = greedy(submod_func, U, d)

    print("f-values:", f_values)

    # Plot
    plot_objective_per_iteration(f_values)