import numpy as np
import math

def poch(a, j):
    """
    Computes the Pochhammer symbol (a)_j = a (a+1) ... (a+j-1) for j>=1 and (a)_0 = 1.
    """
    result = 1.0
    for i in range(j):
        result *= (a + i)
    return result

def univariate_hahn(n, x, alpha, beta, N_local):
    """
    Computes the univariate Hahn polynomial Qₙ(x; α, β, N_local) using its finite series definition:
    
      Qₙ(x; α, β, N_local) = ∑_{j=0}^{n} [(-n)_j (-x)_j (n+α+β+1)_j] / [ (α+1)_j (-N_local+1)_j j! ].
      
    Here the Pochhammer symbol (a)_j is computed via the helper function `poch`.
    """
    s = 0.0
    for j in range(n+1):
        term = (poch(-n, j) * poch(-x, j) * poch(n + alpha + beta + 1, j)) \
               / (poch(alpha + 1, j) * poch(-N_local + 1, j) * math.factorial(j) + 1e-5)
        s += term
    return s

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
      
    where L = ∑ lᵢ and a_[k] is the falling factorial.
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
    
      π(x) = (∏₍ᵢ₌₁₎ᵈ (lᵢ choose xᵢ)) / ( (∑₍ᵢ₌₁₎ᵈ lᵢ choose N) ).
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
    
    Parameters:
      multi_index : Tuple (m₁, …, m₍d₋₁₎) (with |m| = m₁+…+m₍d₋₁₎ = n).
      state       : Tuple (x₁, …, x_d) with ∑ xᵢ = N. Only the first d-1 coordinates are used.
      l_values    : List [l₁, …, l_d].
      N           : Total number of particles.
      
    The definition used is:
    
      Qₘ(x; N, l_values) = ∏₍ᵢ₌₁₎^(d₋₁) Q₍mᵢ₎(xᵢ; αᵢ, βᵢ, Nᵢ),
      
    where:
      αᵢ = lᵢ - 1,
      βᵢ = (∑ⱼ₌ᵢ₊₁ᵈ lⱼ) - 1,
      Nᵢ = N - (x₁ + … + x₍ᵢ₋₁₎).
    """
    d = len(l_values)
    prod = 1.0
    s = 0  # cumulative sum of x₁,...,x₍ᵢ₋₁₎
    for i in range(d - 1):
        alpha_i = l_values[i] - 1
        beta_i = sum(l_values[i+1:]) - 1
        N_i = N - s
        x_i = state[i]
        prod *= univariate_hahn(multi_index[i], x_i, alpha_i, beta_i, N_i)
        s += x_i
    return prod

def MC_generation(N, d, l_values, s):
    """
    Generates a reversible Markov chain with transition matrix P using
    the spectral decomposition based on the real multivariate Hahn polynomials.
    
    The state space is:
         X = { x in ℕᵈ : ∑ xᵢ = N }.
         
    The stationary distribution is given by the multivariate hypergeometric:
         π(x) = (∏₍ᵢ₌₁₎ᵈ (lᵢ choose xᵢ)) / ( (∑₍ᵢ₌₁₎ᵈ lᵢ choose N) ).
         
    The spectral expansion is:
         P(x,y) = π(y) · ∑ₙ₌₀ᴺ βₙ · [∑_{|m|=n} Qₘ(x;N,l_values) Qₘ(y;N,l_values)],
         
    where the eigenvalues βₙ are computed as above.
    
    Parameters:
      N       : Total number of particles.
      d       : Number of types/dimensions.
      l_values: List of parameters [l₁, ..., l_d] (with ∑ lᵢ ≥ N).
      s       : Size parameter for eigenvalues (0 ≤ s ≤ N).
      
    Returns:
      (pi, P) where:
         - pi is a 1D numpy array containing the stationary distribution.
         - P is the transition matrix (2D numpy array).
    """
    # 1. Generate the state space.
    state_space = get_state_space(N, d)
    num_states = len(state_space)
    
    # 2. Compute the stationary distribution.
    pi = compute_stationary_distribution(state_space, l_values, N)
    
    # 3. Compute eigenvalues using the provided formula.
    eigenvalues = compute_eigenvalues(N, s, l_values)
    
    # 4. Build the transition matrix via the spectral decomposition.
    # For each degree n, sum over all multi-indices m in ℕ^(d-1) with |m| = n.
    P = np.zeros((num_states, num_states))
    d_minus_1 = d - 1
    for i, x in enumerate(state_space):
        for j, y in enumerate(state_space):
            spectral_sum = 0.0
            for n in range(N + 1):
                sum_m = 0.0
                multi_indices = get_multi_indices(n, d_minus_1)
                for m in multi_indices:
                    qx = multivariate_hahn(m, x, l_values, N)
                    qy = multivariate_hahn(m, y, l_values, N)
                    sum_m += qx * qy
                spectral_sum += eigenvalues[n] * sum_m
            P[i, j] = spectral_sum * pi[j]
    
    # Renormalize rows to correct for any numerical drift.
    P = P / P.sum(axis=1, keepdims=True)
    return pi, P

# Example usage:
if __name__ == "__main__":
    # Parameters:
    N = 8         # Total number of particles (kept small for state-space size)
    d = 15        # Number of dimensions/types
    # l_values must satisfy ∑ lᵢ ≥ N; here we take all lᵢ = 5.
    l_values = [5] * d  
    # Set the size parameter s (choose any integer 0 ≤ s ≤ N). For instance, s = 1.
    s = 1
    
    # Generate the Markov chain.
    pi, P = MC_generation(N, d, l_values, s)
    
    # For demonstration, print some information.
    state_space = get_state_space(N, d)
    print("Number of states in the space:", len(state_space))
    print("Stationary distribution π (first 10 states):", pi[:10])
    print("Transition matrix P shape:", P.shape)
