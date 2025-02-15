# Submodularity optimization of the entropy rate

We use submodularity maximization on entropy rate to find an optimal subset Markov chain (with the largest entropy rate) with cardinality constraints.

The following map:
$$S \mapsto H(P^{(S)})$$
is submodular but generally not monotonically non-decreasing, so the greedy algorithm used is a heuristic algorithm but not near-optimal. Hence, we also consider the following map:
$$S \mapsto H(P^{(S)}) = H(\pi^{(S)} \boxtimes P^{(S)}) - H(\pi^{(S)}),$$
which is a monotonically non-decreasing submodular function minus a modular function if we assume that $\pi$ is of product form, i.e. $\pi = \otimes_{i=1}^d \pi_i$, and we can use distorted greedy algorithm for this objective function.

![Distorted Greedy Algorithm](/assets/distgrdy.png)

We consider the following approaches:
* Approach 1: Heuristic greedy algorithm, see `main_approach_1.py` and `main_approach_1_gpu.py`.
* Approach 2: Distorted greedy algorithm, requiring $\pi$ to be of product form, see `main_approach_2.py` and `main_approach_2_gpu.py`.
* Approach 3: Distorted greedy algorithm, choose $\beta = 0$, see `main_approach_3_gpu.py`.

The following is the visualization these algorithms, in which case we aim to choose a subset Markov chain with at most 4 dimensions out of a 15-dimensional Markov chain.

**Approach 1**
![visualization](/assets/sample_paths_entropyrate_1.png)

**Approach 2**
![visualization](/assets/sample_paths_entropyrate_2.png)

**Approach 3**
![visualization](/assets/sample_paths_entropyrate_3.png)

# References
* [1] Harshaw, C., Feldman, M., Ward, J., & Karbasi, A. (2019). Submodular maximization beyond non-negativity: Guarantees, fast algorithms, and applications. In International Conference on Machine Learning (pp. 2634-2643). PMLR.