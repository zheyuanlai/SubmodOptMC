# This file contains several methods of submodular maximization with cardinality constraints.
import numpy as np # type: ignore

def greedy(f, X, k):
    '''
    Greedy algorithm for submodular maximization with cardinality constraints.

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
        gain, elem = max(gains)
        S.add(elem)
    return S

def lazy_greedy(f, X, k):
    '''
    Lazy greedy algorithm for submodular maximization with cardinality constraints.

    Parameters:
        f (function): The submodular function.
        X (set): The ground set.
        k (int): The cardinality constraint.

    Returns:
        set: The selected subset.
    '''
    S = set()
    gains = [(f(S.union({e})) - f(S), e) for e in X - S]
    gain, elem = max(gains)
    S.add(elem)
    for _ in range(k - 1):
        gains = [(f(S.union({e})) - f(S), e) for e in X - S]
        gain, elem = max(gains)
        S.add(elem)
    return S

def stochastic_greedy(f, X, k, epsilon=0.1):
    '''
    Stochastic greedy algorithm for submodular maximization with cardinality constraints.

    Parameters:
        f (function): The submodular function.
        X (set): The ground set.
        k (int): The cardinality constraint.
        T (int): The number of iterations.
        epsilon (float): Error tolerance parameter, default is 0.1.

    Returns:
        set: The selected subset.
    '''
    S = set()
    n = len(X)
    s = int((n / k) * np.log(1 / epsilon))

    for _ in range(k):
        R = np.random.choice(list(X - S), size=min(s, len(X - S)), replace=False)
        gains = [(f(S.union({e})) - f(S), e) for e in R]
        probs = [gain / sum(g for g, _ in gains) for gain, _ in gains]
        elem = np.random.choice([e for _, e in gains], p=probs)
        S.add(elem)
    return S

# Example function of the greedy algorithm
def example_submodular_function(S):
    # Example submodular function: f(S) = sqrt(|S|)
    return np.sqrt(sum(S))

if __name__ == "__main__":
    # Example ground set and cardinality constraint
    X = {1, 2, 3, 4, 5}
    k = 3

    # Example usage of the greedy algorithm
    selected_set = greedy(example_submodular_function, X, k)
    print("Selected subset (greedy):", selected_set)

    # Example usage of the lazy greedy algorithm
    selected_set = lazy_greedy(example_submodular_function, X, k)
    print("Selected subset (lazy greedy):", selected_set)

    # Example usage of the stochastic greedy algorithm
    selected_set = stochastic_greedy(example_submodular_function, X, k)
    print("Selected subset (stochastic greedy):", selected_set)