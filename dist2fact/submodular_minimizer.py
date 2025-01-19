'''Submodular minimizer using Lovasz's Extension.'''
import numpy as np # type: ignore

def lovasz_extension(F, w):
    """
    Compute the Lovász extension of a submodular function F at point w.
    """
    n = len(w)
    sorted_indices = np.argsort(-w)
    f = 0
    prev_F = 0
    for k in range(n):
        S = sorted_indices[:k+1]
        curr_F = F(S)
        f += w[sorted_indices[k]] * (curr_F - prev_F)
        prev_F = curr_F
    return f

def subgradient(F, w):
    """
    Compute a subgradient of the Lovász extension at point w.
    """
    n = len(w)
    sorted_indices = np.argsort(-w)
    s = np.zeros(n)
    prev_F = 0
    for k in range(n):
        S = sorted_indices[:k+1]
        curr_F = F(S)
        s[sorted_indices[k]] = curr_F - prev_F
        prev_F = curr_F
    return s

def projected_subgradient_descent(F, n, max_iter=1000, epsilon=1e-6):
    """
    Minimize a submodular function F using projected subgradient descent.
    """
    w = np.zeros(n)
    for t in range(1, max_iter + 1):
        eta_t = 1 / np.sqrt(t)
        s_t = subgradient(F, w)
        w_new = w - eta_t * s_t
        w_new = np.clip(w_new, 0, 1)
        if np.linalg.norm(w_new - w) < epsilon:
            break
        w = w_new
    A_star = np.where(w >= 0.5)[0]
    return A_star

'''Submodular minimizer using Greedy algorithm.'''
def greedy(f, X, k):
    '''
    Greedy algorithm for submodular minimization with cardinality constraints.

    Parameters:
        f (function): The submodular function.
        X (set): The ground set.
        k (int): The cardinality constraint.

    Returns:
        set: The selected subset.
    '''
    S = set()
    for _ in range(k):
        gains = [(f(S.union({e})) - f(S), e) for e in X - S]
        gain, elem = min(gains)
        S.add(elem)
    return S

# Example usage
def example_submodular_function(S):
    return len(S)

if __name__ == "__main__":
    n = 5
    A_star = projected_subgradient_descent(example_submodular_function, n)
    print("Minimizer subset:", A_star)