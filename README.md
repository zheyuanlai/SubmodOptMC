<div align="center">
<h3>Information-theoretic subset selection of multivariate Markov chains via submodular optimization</h3>

[Zheyuan Lai](https://zheyuanlai.github.io)* and [Michael C.H. Choi](https://mchchoi.github.io)†

*: Department of Statistics and Data Science, National University of Singapore, Singapore; Email: zheyuan_lai@u.nus.edu

†: Department of Statistics and Data Science and Yale-NUS College, National University of Singapore, Singapore; Email: mchchoi@nus.edu.sg, corresponding author

💻 [[GitHub]](https://github.com/zheyuanlai/SubmodOptMC)
</div>

<details>
<summary>📄 Abstract</summary>
Given a finite ergodic multivariate Markov chain on a product state space, this paper seeks to find an optimal projected Markov chain with respect to information-theoretic objective functions and subjected to suitable cardinality constraints. The objectives that we consider include the classical entropy rate, information-theoretic distance to factorizability, distance to independence as well as distance to stationarity. These combinatorial optimization problems can be broadly understood as best subset selection in the context of multivariate Markov chains. As these set functions are submodular (or supermodular), we design greedy-based algorithms to solve for these problems along with theoretical guarantees. We extend our analysis to the multivariate setting, and propose a generalized distorted greedy algorithm for maximizing $k$-submodular functions subjected to cardinality constraints, which is of independent interests. We provide numerical experiments of multivariate Markov chains associated with the Bernoulli-Laplace model and the Curie-Weiss model that demonstrate the performance of the greedy algorithms.
</details>

## 👋 Overview
This project aims to addresses the challenges regarding subset selection in multivariate Markov chains by integrating information theory with submodular optimization. Motivated by the need to efficiently capture and quantify dependencies in high-dimensional stochastic systems, our framework identifies optimal subsets subject to some cardinality constraints through principled information-theoretic criteria in the Markov chain theory. By employing advanced greedy algorithms with provable theoretical guarantees, this framework ensures scalable and efficient solutions for complex stochastic systems.

## 📁 Code Structure

```
/SubmodOptMC
├── dist2fact/              # Distance to factorizability (Section 4 of the paper)
├── dist2fact_fixed/        # Distance to factorizability over a fixed set (Section 7 of the paper)
├── dist2indp/              # Distance to independence (Section 5 of the paper)
├── dist2stat/              # Distance to stationarity (Section 6 of the paper)
├── entropy_rate/           # Entropy rate (Section 3 of the paper)
├── results/                # Numerical experiment results
├── README.md               # Project README file
└── requirements.txt        # Python dependencies
```

## 🔬 Numerical Experiments
We consider the multivariate Markov chains associated with the Bernoulli-Laplace level model [7] and the Curie-Weiss model [1] (see Section 8.1 and Section 8.2 of our paper). We examine the performance of the following algorithms on both models:

* Heuristic greedy algorithm (see Section 4 of [11])
* Distorted greedy algorithm [5] (see Algorithm 2 of our paper)
* Generalized distorted greedy algorithm (see Algorithm 3 of our paper)
* Batch greedy algorithm [6] (see Algorithm 4 of our paper)

## ⌨️ Usage
To reproduce the numerical experiments presented in the manuscript:
1. Set up a new Conda environment:
```bash
conda create -n submodular python=3.10 -y
conda activate submodular
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Navigate to the relevant folder (e.g., `entropy_rate/`) and run the experiment scripts.

## 📊 Results
The `results/Bernoulli-Laplace/` and `results/Curie-Weiss/` folders contain:
* Plots comparing the performance of different algorithms.
* Logs and output files from the numerical experiments.

## 📚 References

1. Anton Bovier and Frank Den Hollander. *Metastability: a potential-theoretic approach*, volume 351. Springer, 2016.

2. Michael C.H. Choi, Youjia Wang, and Geoffrey Wolfer. Geometry and factorization of multivariate markov chains with applications to the swapping algorithm. *arXiv preprint arXiv:2404.12589*, 2024.

3. Alina Ene and Huy Nguyen. Streaming algorithm for monotone k-submodular maximization with cardinality constraints. In *International Conference on Machine Learning*, pages 5944–5967. PMLR, 2022.

4. Uriel Feige, Vahab S Mirrokni, and Jan Vondrák. Maximizing non-monotone submodular functions. *SIAM Journal on Computing*, 40(4):1133–1153, 2011.

5. Chris Harshaw, Moran Feldman, Justin Ward, and Amin Karbasi. Submodular maximization beyond non-negativity: Guarantees, fast algorithms, and applications. In Kamalika Chaudhuri and Ruslan Salakhutdinov, editors, *Proceedings of the 36th International Conference on Machine Learning*, volume 97 of *Proceedings of Machine Learning Research*, pages 2634–2643. PMLR, 09–15 Jun 2019.

6. Jayanth Jagalur-Mohan and Youssef Marzouk. Batch greedy maximization of non-submodular functions: Guarantees and applications to experimental design. *Journal of Machine Learning Research*, 22(252):1–62, 2021.

7. Kshitij Khare and Hua Zhou. Rates of convergence of some multivariate markov chains with polynomial eigenfunctions. 2009.

8. Bernhard H Korte, Jens Vygen, B Korte, and J Vygen. *Combinatorial optimization*, volume 1. Springer, 2011.

9. Jon Lee, Maxim Sviridenko, and Jan Vondrák. Submodular maximization over multiple matroids via generalized exchange properties. *Mathematics of Operations Research*, 35(4):795–806, 2010.

10. David A Levin and Yuval Peres. *Markov chains and mixing times*, volume 107. American Mathematical Soc., 2017.

11. George L Nemhauser, Laurence A Wolsey, and Marshall L Fisher. An analysis of approximations for maximizing submodular set functions—i. *Mathematical programming*, 14:265–294, 1978.

12. Yury Polyanskiy and Yihong Wu. *Information Theory: From Coding to Learning*. Cambridge University Press, 2025.

13. Justin Ward and Stanislav Zivný. Maximizing k-submodular functions and beyond. *CoRR*, abs/1409.1399, 2014.