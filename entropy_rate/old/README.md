# Submodularity optimization of the entropy rate

We use submodularity maximization on entropy rate to find an optimal subset Markov chain (with the largest entropy rate) with cardinality constraints.

For the generation of Markov chain, we generate a generalized Bernoulli–Laplace model as detailed in Section 4.2 of [2].

The following map:
$$S \mapsto H(P^{(S)})$$
is submodular but generally not monotonically non-decreasing, so the greedy algorithm used is a heuristic algorithm (see Section 4 of [1]) but not near-optimal. 

We apply the following distorted greedy algorithm to solve this problem with a lower bound, which is introduced in [3], see Corollary 2.2 of the manuscript for the detailed approach.

We consider the following approaches:
* Approach 1: Heuristic greedy algorithm, see `greedy.py`.
* Approach 2: Distorted greedy algorithm, choose $\beta = 0$, see `distorted_greedy.py`.

We also want to maximize the entropy rate of the tensorized keep-$S_i$-in matrices $$H(\otimes_{i=1}^k P^{(S_i)})$$, specifically, we aim to consider the following maximization problem:
$$\mathbf{S} = (S_1, \ldots, S_k) \mapsto H(\otimes_{i=1}^k P^{(S_i)})$$.

We first apply Theorem 1.12 to give a $k$-submodular function $g$ and a modular function $c$, then we apply the generalized distorted greedy algorithm, see Corollary 2.5 for the detailed approach.

# References
* [1] Nemhauser, G.L., Wolsey, L.A. & Fisher, M.L. An analysis of approximations for maximizing submodular set functions—I. Mathematical Programming 14, 265–294 (1978). https://doi.org/10.1007/BF01588971
* [2] Kshitij Khare. Hua Zhou. "Rates of convergence of some multivariate Markov chains with polynomial eigenfunctions." Ann. Appl. Probab. 19 (2) 737 - 777, April 2009. https://doi.org/10.1214/08-AAP562
* [3] Harshaw, C., Feldman, M., Ward, J., & Karbasi, A. (2019). Submodular maximization beyond non-negativity: Guarantees, fast algorithms, and applications. In International Conference on Machine Learning (pp. 2634-2643). PMLR.