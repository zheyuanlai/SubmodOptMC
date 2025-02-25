# Submodular optimization of KL divergence

We use submodular minimization on the KL divergence between original transition probability matrix P and the outer product of keep-S-in and leave-S-out matrices, with $S$ subject to a cardinality constraint $|S|=k$. We propose a heuristic greedy algorithm to do so, see `main_greedy_minimization.py`.

We are also interested in submodular maximization on the KL divergence between original transition probability matrix P and the outer product of keep-S-in and leave-S-out matrices, subject to a cardinality constraint $|S| \leq k$. We apply Proposition 14.18 of [2], get monotonically non-decreasing function $g(S) = D(P \| P^{(S)} \otimes P^{(-S)}) + \sum_{e \in S} D(P \| P^{(U \backslash \{e\})} \otimes P^{(e)})$. Since $\sum_{e \in S} D(P \| P^{(U \backslash \{e\})} \otimes P^{(e)})$ is modular, we have $D(P \| P^{(S)} \otimes P^{(-S)}) = g(S) - c(S)$, in which case we can apply the distorted greedy algorithm in [1], see `main_distorted_greedy_maximization.py`.

The following is the visualization these algorithms, in which case we aim to choose a subset Markov chain with at most 4 dimensions out of a 13-dimensional Markov chain.

![Distance to factorizability](/assets/sample_paths_dist_grdy_dist2fact.png)

# References
* [1] Harshaw, C., Feldman, M., Ward, J., & Karbasi, A. (2019). Submodular maximization beyond non-negativity: Guarantees, fast algorithms, and applications. In International Conference on Machine Learning (pp. 2634-2643). PMLR.
* [2] Korte, B. H., Vygen, J., Korte, B., & Vygen, J. (2011). Combinatorial optimization (Vol. 1, pp. 1-595). Berlin: Springer.