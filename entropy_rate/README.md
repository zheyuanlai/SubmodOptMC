# Submodularity optimization of the entropy rate

We use submodularity maximization on entropy rate to find an optimal subset Markov chain (with the largest entropy rate) with cardinality constraints.

The following map:
$$S \mapsto H(P^{(S)})$$
is submodular but generally not monotonically non-decreasing, so the greedy algorithm used is a heuristic algorithm but not near-optimal. Hence, we also consider the following map:
$$S \mapsto H(P^{(S)}) = H(\pi^{(S)} \boxtimes P^{(S)}) - H(\pi^{(S)}),$$
which is a monotonically non-decreasing submodular function minus a modular function if we assume that $\pi$ is of product form, i.e. $\pi = \otimes_{i=1}^d \pi_i$, and we can use distorted greedy algorithm for this objective function.

![Distorted Greedy Algorithm](/assets/distgrdy.png)