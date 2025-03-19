import torch
import numpy as np
import math
import matplotlib.pyplot as plt
from itertools import product

# ---- Device selection ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------
# JIT-compiled helper functions
# -----------------------
@torch.jit.script
def poch_torch(a: float, j: int) -> float:
    result = 1.0
    for i in range(j):
        result *= a + i
    return result

@torch.jit.script
def univariate_hahn_torch(n: int, x: float, alpha: float, beta: float, N_local: float) -> float:
    s_val = 0.0
    eps = 1e-10
    for j in range(n + 1):
        num = poch_torch(float(-n), j) * poch_torch(-x, j) * poch_torch(float(n + alpha + beta + 1), j)
        den = poch_torch(alpha + 1.0, j) * (poch_torch(float(-N_local + 1), j) + eps) * math.factorial(j)
        if den == 0: continue
        s_val += num / den
    return s_val

# -----------------------
# Precompute lookup tables
# -----------------------
def precompute_lookup_tables(l_values, N, max_m):
    """
    For each free coordinate i (i=0,..., d-2), precompute a lookup table of shape
       (max_m+1, 2, N+1)
    where for a given m (0 <= m <= max_m), x (0 or 1), and N_val in {0,...,N},
       lookup[i][m, x, N_val] = Q_m(x; alpha_i, beta_i, N_val)
    with alpha_i = l_values[i]-1 and beta_i = (sum(l_values[i+1:]) - 1).
    """
    d = len(l_values)
    lookup_tables = []
    for i in range(d - 1):
        alpha_i = l_values[i] - 1
        beta_i = sum(l_values[i+1:]) - 1
        # Table shape: (max_m+1, 2, N+1)
        table = torch.empty((max_m+1, 2, N+1), dtype=torch.float32, device=device)
        for m in range(max_m+1):
            for x in [0, 1]:
                for N_val in range(N+1):
                    table[m, x, N_val] = univariate_hahn_torch(m, float(x), float(alpha_i), float(beta_i), float(N_val))
        lookup_tables.append(table)
    return lookup_tables

# -----------------------
# State space generation
# -----------------------
def get_product_state_space(num_free):
    """
    Generate the full product state space for free coordinates (each in {0,1})
    as a torch tensor of shape (M, num_free) (dtype=int64).
    """
    grid = torch.tensor([0, 1], dtype=torch.int64, device=device)
    grids = [grid for _ in range(num_free)]
    return torch.cartesian_prod(*grids)  # shape: (2^(num_free), num_free)

# -----------------------
# Stationary distribution computation
# -----------------------
def stat_dist_MC_generation(state_space, l_values, N):
    """
    Given the free state space (tensor of shape (M, d-1) of type int64), compute the stationary distribution.
    Form full state by appending x_d = N - sum(x).
    """
    d = len(l_values)
    M = state_space.shape[0]
    free = state_space.to(torch.int64)
    last = (N - free.sum(dim=1, keepdim=True)).to(torch.int64)
    full = torch.cat([free, last], dim=1)  # shape: (M, d)
    l_total = sum(l_values)
    denom = math.comb(l_total, N)
    pi_vals = torch.empty(M, dtype=torch.float32, device=device)
    full_np = full.cpu().numpy()
    for i, row in enumerate(full_np):
        prod_val = 1
        for xi, li in zip(row, l_values):
            prod_val *= math.comb(li, xi)
        pi_vals[i] = prod_val / denom
    pi_vals = pi_vals / pi_vals.sum()
    return pi_vals

# -----------------------
# Vectorized multivariate Hahn evaluation
# -----------------------
def compute_Mn(state_space, multi_indices, lookup_tables, N):
    """
    Compute M_n as a tensor of shape (M, K) where:
      M_n[i,j] = ∏_{i=0}^{d-2} lookup_tables[i][ m[i], x_i, N_i ]
    for each full state and each multi-index.
    
    Parameters:
       state_space: tensor of shape (M, d-1) (free coordinates, int64)
       multi_indices: tensor of shape (K, d-1) (int64)
       lookup_tables: list of (d-1) tensors, each of shape (max_m+1, 2, N+1)
       N: total particle count (scalar)
    
    For each state row and for coordinate i, 
      x_i = state_space[:, i] (0 or 1)
      N_i = N - cumulative sum of free coordinates before i.
    """
    M, d_minus1 = state_space.shape
    K = multi_indices.shape[0]
    # Compute cumulative sum along each free state row:
    cumsum = torch.cumsum(state_space.to(torch.int64), dim=1)  # shape (M, d-1)
    # For coordinate i, define N_i = N - (if i==0 then 0 else cumsum[:, i-1])
    Ni = torch.zeros((M, d_minus1), dtype=torch.int64, device=device)
    Ni[:, 0] = N  # when i==0, cumulative sum is 0
    if d_minus1 > 1:
        Ni[:, 1:] = N - cumsum[:, :-1]
    # Prepare an output tensor for M_n:
    M_n = torch.ones((M, K), dtype=torch.float32, device=device)
    # For each coordinate i (from 0 to d-2), multiply in the corresponding factor.
    for i in range(d_minus1):
        # For this coordinate, we want to index lookup_tables[i] with:
        #   m_indices: multi_indices[:, i] of shape (K,) -> expand to (M,K)
        #   x_indices: state_space[:, i] of shape (M,) -> expand to (M,K)
        #   Ni: computed from above, shape (M,) for coordinate i -> (M,K)
        m_idx = multi_indices[:, i].unsqueeze(0).expand(M, K)  # shape (M, K)
        x_idx = state_space[:, i].unsqueeze(1).expand(M, K)     # shape (M, K)
        Ni_idx = Ni[:, i].unsqueeze(1).expand(M, K)             # shape (M, K)
        # Use advanced indexing on lookup_tables[i] of shape (N+1, 2, N+1).
        table = lookup_tables[i]
        # All indices must be of type torch.long.
        m_idx = m_idx.to(torch.long)
        x_idx = x_idx.to(torch.long)
        Ni_idx = Ni_idx.to(torch.long)
        # Now, table has shape (max_m+1, 2, N+1). Use:
        # value = table[m_idx, x_idx, Ni_idx]  (all same shape (M,K))
        value = table[m_idx, x_idx, Ni_idx]
        M_n = M_n * value
    return M_n

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

def get_multi_indices_tensor(total, length):
    """
    Generate all tuples of nonnegative integers of given length summing to total.
    Returns a tensor of shape (num_indices, length) of type int64.
    """
    # Use recursive Python function to generate list of tuples, then convert to tensor.
    result = []
    def recurse(tleft, dim, partial):
        if dim == 1:
            result.append(partial + [tleft])
            return
        for v in range(tleft + 1):
            recurse(tleft - v, dim - 1, partial + [v])
    recurse(total, length, [])
    return torch.tensor(result, dtype=torch.int64, device=device)

# -----------------------
# Markov Chain Generation (Vectorized Version)
# -----------------------
def torch_MC_generation_vec(N, d, l_values, s, product_form=True):
    """
    Generate the Bernoulli–Laplace chain using vectorized operations.
    Returns state_space (tensor of free coordinates), stationary distribution pi,
    and transition matrix P.
    """
    if product_form:
        num_free = d - 1
        state_space = get_product_state_space(num_free)  # (M, d-1) int64
    else:
        raise NotImplementedError("Only product state space is implemented in vectorized version.")
    M = state_space.shape[0]
    pi = stat_dist_MC_generation(state_space, l_values, N)  # (M,)
    eigen_tensor = compute_eigenvalues(N, s, l_values)      # shape (N+1,)

    # Precompute lookup tables for free coordinates (i=0,...,d-2)
    max_m = N  # maximum value for m in multi-indices
    lookup_tables = precompute_lookup_tables(l_values, N, max_m)

    M_n_list = []
    for n in range(N + 1):
        multi_indices = get_multi_indices_tensor(n, d - 1)  # (K, d-1), int64
        M_n = compute_Mn(state_space, multi_indices, lookup_tables, N)  # (M, K)
        M_n_list.append(M_n)
    A = torch.zeros((M, M), dtype=torch.float32, device=device)
    for n in range(N + 1):
        beta_n = eigen_tensor[n]
        M_n = M_n_list[n]
        A_n = torch.matmul(M_n, M_n.t())
        A += beta_n * A_n
    A = torch.clamp(A, min=0.0)
    P = A * pi.unsqueeze(0)  # weight columns by pi
    P = P / (P.sum(dim=1, keepdim=True) + 1e-15)
    P = torch.clamp(P, min=0.0)
    P = P / (P.sum(dim=1, keepdim=True) + 1e-15)
    return state_space, pi, P

# -----------------------
# Aggregation and Entropy Rate (similar to before)
# -----------------------
def keep_S_in_mat(P, state_space, pi, S):
    M = state_space.shape[0]
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
    #d = len(state_space[0])
    Sbar = set(range(state_space.shape[1])) - set(S)
    return keep_S_in_mat(P, state_space, pi, Sbar)

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

        S[best_j].add(best_e)

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
    N = 10
    d = N + 1
    l_values = [1]*(d-1) + [N]
    s = 1

    k = 3
    V = [
        set(range(0, 4)),
        set(range(4, 7)),
        set(range(7, 10))
    ]

    # Generate Markov chain with vectorized operations:
    state_space, pi, P = torch_MC_generation_vec(N, d, l_values, s, product_form=True)
    print(f"Generated chain with product state space of dimension {d-1} (total states = {state_space.shape[0]})")

    def dist2stat(S):
        _, piS, PS = keep_S_in_mat(P, state_space, pi, S)
        _, piSbar, PSbar = leave_S_out_mat(P, state_space, pi, S)
        return KL_divergence_gpu(piS, PS, piS.repeat(len(piS), 1))

    def f(S):
        val = 0.0
        for coord in S:
            val += dist2stat(coord)
        return -val
    
    def c(S):
        val = 0.0
        for j, coord in enumerate(S):
            for elem in coord:
                Vj_minus_elem = set(V[j]) - {elem}
                _, pi_e, P_e = keep_S_in_mat(P, state_space, pi, {elem})
                _, pi_V, P_V = keep_S_in_mat(P, state_space, pi, V[j])
                _, pi_V_minus_elem, P_V_minus_elem = keep_S_in_mat(P, state_space, pi, Vj_minus_elem)
                val += KL_divergence_gpu(pi_V, P_V, compute_outer_product_gpu(P_e, P_V_minus_elem))
                val += KL_divergence_gpu(pi_e, P_e, pi_e.repeat(len(pi_e), 1))
        return val
    
    def g(S):
        return f(S) + c(S)
    
    for m in range(1, d):
        S = distorted_greedy_k_submod(g, c, V, m, k)
        print(f"m={m}; Subset chosen: {S}; Value: {-f(S)}")