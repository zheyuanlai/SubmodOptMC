import torch
import numpy as np
import math

# -------------------------------
# Helper functions (CPU side)
# -------------------------------

def poch(a, j):
    """
    Computes the Pochhammer symbol (a)_j = a (a+1) ... (a+j-1) for j>=1,
    with (a)_0 = 1.
    """
    result = 1.0
    for i in range(j):
        result *= (a + i)
    return result

def univariate_hahn(n, x, alpha, beta, N_local):
    """
    Computes the univariate Hahn polynomial Qₙ(x; α, β, N_local) using its finite series definition:
    
      Qₙ(x; α, β, N_local) = ∑_{j=0}^{n} [(-n)_j (-x)_j (n+α+β+1)_j] / [ (α+1)_j (-N_local+1)_j j! ].
    
    (Here we add a small epsilon in the denominator to avoid division-by-zero.)
    """
    s_val = 0.0
    eps = 1e-10
    for j in range(n+1):
        num = poch(-n, j) * poch(-x, j) * poch(n + alpha + beta + 1, j)
        den = (poch(alpha + 1, j) * poch(-N_local + 1, j) * math.factorial(j) + eps)
        s_val += num / den
    return s_val

def falling_factorial(a, k):
    """Computes the falling factorial a_[k] = a (a-1) ... (a-k+1)."""
    result = 1
    for i in range(k):
        result *= (a - i)
    return result

def compute_eigenvalues(N, s, l_values):
    """
    Compute the eigenvalues βₙ for n = 0,...,N for the Bernoulli–Laplace model,
    using the formula:
    
      βₙ = ∑ₖ₌₀ⁿ (n choose k) * ((N-s)_[n-k] * s_[k]) / (N_[n-k] * (L-N)_[k]),
      
    where L = ∑ lᵢ and a_[k] denotes the falling factorial.
    """
    l_total = sum(l_values)
    eigenvalues = []
    for n in range(N + 1):
        sum_term = 0.0
        for k in range(n + 1):
            num = math.comb(n, k) * falling_factorial(N - s, n - k) * falling_factorial(s, k)
            den = falling_factorial(N, n - k) * falling_factorial(l_total - N, k)
            sum_term += num / den
        eigenvalues.append(sum_term)
    return eigenvalues

def get_state_space(N, d):
    """
    Recursively generate the state space:
      X = { x in ℕᵈ : sum(x) = N }.
    Returns a list of tuples.
    """
    if d == 1:
        return [(N,)]
    state_space = []
    for x0 in range(N + 1):
        for tail in get_state_space(N - x0, d - 1):
            state_space.append((x0,) + tail)
    return state_space

def compute_stationary_distribution(state_space, l_values, N):
    """
    Computes the stationary distribution π for each state x as:
    
      π(x) = (∏ (lᵢ choose xᵢ)) / ((∑ lᵢ choose N)).
    Returns a NumPy array.
    """
    total = sum(l_values)
    denominator = math.comb(total, N)
    pi = []
    for state in state_space:
        numerator = 1
        for x, l in zip(state, l_values):
            numerator *= math.comb(l, x)
        pi.append(numerator / denominator)
    pi = np.array(pi)
    return pi / pi.sum()

def get_multi_indices(total, length):
    """
    Generate all tuples (m₁, ..., m_length) of nonnegative integers that sum to 'total'.
    """
    if length == 1:
        return [(total,)]
    indices = []
    for i in range(total + 1):
        for tail in get_multi_indices(total - i, length - 1):
            indices.append((i,) + tail)
    return indices

def multivariate_hahn(multi_index, state, l_values, N):
    """
    Computes the multivariate Hahn polynomial Qₘ(x; N, l_values) for a given multi-index.
    
    For state x = (x₁, ..., x_d) (with sum(x)=N) and multi-index
    m = (m₁, ..., m₍d₋₁₎), we set
      Qₘ(x) = ∏_{i=1}^{d-1} Q_{m_i}(x_i; αᵢ, βᵢ, Nᵢ),
    with αᵢ = lᵢ - 1, βᵢ = (∑_{j=i+1}^d l_j) - 1, and Nᵢ = N - (x₁+…+x₍i-1₎).
    """
    d = len(l_values)
    prod = 1.0
    s_val = 0  # cumulative sum x₁,...,x_{i-1}
    for i in range(d - 1):
        alpha_i = l_values[i] - 1
        beta_i = sum(l_values[i+1:]) - 1
        N_i = N - s_val
        x_i = state[i]
        prod *= univariate_hahn(multi_index[i], x_i, alpha_i, beta_i, N_i)
        s_val += x_i
    return prod

# -------------------------------
# PyTorch (GPU) version
# -------------------------------

def torch_MC_generation(N, d, l_values, s, device):
    """
    Generates the Markov chain transition matrix P using the spectral decomposition
    with real multivariate Hahn polynomials, and moves heavy computations to the GPU.
    
    Steps:
      1. Compute the state space and stationary distribution.
      2. For each degree n (0<= n <=N), precompute for every state x the values
         Qₘ(x) for all multi-indices m of length (d-1) with sum m = n.
      3. For each degree, form the matrix Mₙ (shape: [num_states, num_m_indices]),
         then compute Aₙ = Mₙ Mₙᵀ.
      4. Sum A = ∑ₙ βₙ Aₙ, and set P(x,y) = π(y)*A(x,y), then renormalize rows.
    
    Returns:
      pi (torch tensor, shape [num_states]) and P (torch tensor, shape [num_states, num_states])
      on the given device.
    """
    # 1. State space and stationary distribution.
    state_space = get_state_space(N, d)   # list of tuples
    num_states = len(state_space)
    pi_np = compute_stationary_distribution(state_space, l_values, N)  # numpy array
    pi = torch.tensor(pi_np, dtype=torch.float32, device=device)  # shape: [num_states]
    
    # 2. Eigenvalues (small list) computed on CPU.
    eigenvalues = compute_eigenvalues(N, s, l_values)  # list of length N+1

    # 3. For each degree n, precompute the matrix M_n.
    # We'll store M_n as a tensor of shape [num_states, num_multi_indices].
    M_n_list = []
    for n in range(N+1):
        multi_indices_n = get_multi_indices(n, d - 1)  # list of tuples for this degree
        num_multi = len(multi_indices_n)
        # Create an empty array [num_states, num_multi] to hold Q-values.
        M_n = np.zeros((num_states, num_multi), dtype=np.float32)
        for i, state in enumerate(state_space):
            for j, m in enumerate(multi_indices_n):
                M_n[i, j] = multivariate_hahn(m, state, l_values, N)
        # Convert to torch tensor and move to device.
        M_n_tensor = torch.tensor(M_n, dtype=torch.float32, device=device)
        M_n_list.append(M_n_tensor)
    
    # 4. Build the matrix A = sum_n (beta_n * M_n M_n^T)
    A = torch.zeros((num_states, num_states), dtype=torch.float32, device=device)
    for n in range(N+1):
        beta_n = eigenvalues[n]
        M_n = M_n_list[n]  # shape: [num_states, num_multi]
        A_n = torch.matmul(M_n, M_n.t())  # shape: [num_states, num_states]
        A = A + beta_n * A_n

    # 5. Build the transition matrix: P(x,y) = π(y)*A(x,y)
    P = A * pi.unsqueeze(0)  # Multiply each column y by π(y)
    # Renormalize rows so that each row sums to 1.
    P = P / (P.sum(dim=1, keepdim=True) + 1e-10)
    
    return pi, P

# -------------------------------
# Example usage
# -------------------------------

if __name__ == "__main__":
    # Parameters:
    N = 3         # total number of particles (kept small)
    d = 20        # number of dimensions/types
    l_values = [5] * d   # must satisfy sum(l_values) >= N
    s = 1              # size parameter (0 <= s <= N)
    
    # Set device: use CUDA if available, otherwise MPS if available, else CPU.
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("Using device:", device)
    
    # Generate the Markov chain.
    pi, P = torch_MC_generation(N, d, l_values, s, device)
    
    # For demonstration, bring results back to CPU and print some info.
    pi_cpu = pi.cpu().numpy()
    P_cpu = P.cpu().numpy()
    state_space = get_state_space(N, d)
    print("Number of states in the space:", len(state_space))