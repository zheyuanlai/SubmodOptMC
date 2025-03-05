'''Minimizing a supermodular function is equivalent to maximizing a submodular function.'''

def submodular_greedy(f, X, k):
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

def greedy(f, X, k):
    '''
    Greedy algorithm for supermodular minimization with cardinality constraints.

    Parameters:
        f (function): The supermodular function.
        X (set): The ground set.
        k (int): The cardinality constraint.

    Returns:
        set: The selected subset.
    '''
    return submodular_greedy(lambda x: -f(x), X, k)