# Submodularity optimization of the entropy rate

We use submodularity maximization on entropy rate to find an optimal subset Markov chain (with the largest entropy rate) with cardinality constraints.

The following map:
$$S \mapsto H(P^{(S)})$$
is submodular but generally not monotonically non-decreasing, so the greedy algorithm used is a heuristic algorithm but not near-optimal. Hence, we also consider the following map:
$$S \mapsto H(P^{(S)}) = H(\pi^{(S)} \boxtimes P^{(S)}) - H(\pi^{(S)}),$$
which is a monotonically non-decreasing submodular function minus a modular function if we assume that $\pi$ is of product form, i.e. $\pi = \otimes_{i=1}^d \pi_i$, and we can use distorted greedy algorithm for this objective function.

![Distorted Greedy Algorithm](/assets/distgrdy.png)

The greedy approach is in `main_greedy.py`, and the distorted greedy approach is in `main_distorted_greedy.py`.

The following is the visualization of greedy algorithm, in which case we aim to choose a subset Markov chain with at most 4 dimensions out of a 15-dimensional Markov chain.

![visualization](/assets/sample_paths_grdy_entropy.png)

The following is the visualization of distorted greedy algorithm, in which case we aim to choose a subset Markov chain with at most 4 dimensions out of a 15-dimensional Markov chain.

![visualization](/assets/sample_paths_dist_grdy_entropy.png)

It turns out that in small dimensional cases, although distorted greedy algorithm has an approximation guarantee, it does not necessarily yields the optimal results, since 
$$g(S_m) - c(S_m) \geq (1 - e^{-1}) g(\mathrm{OPT}) - c(\mathrm{OPT}).$$

# References
* [1] Harshaw, C., Feldman, M., Ward, J., & Karbasi, A. (2019). Submodular maximization beyond non-negativity: Guarantees, fast algorithms, and applications. In International Conference on Machine Learning (pp. 2634-2643). PMLR.