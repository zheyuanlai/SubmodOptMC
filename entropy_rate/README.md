# Submodularity optimization of the entropy rate

We use submodularity maximization on entropy rate to find an optimal subset Markov chain with knapsack constraints.

To perform numerical experiments, run the main script:
```
python main.py
```

The script `main.py` also includes testing part, which compares the entropy rates between optimal `keep_S_in` and other non-optimal `keep_S_in` of Markov chains.

The submodularity maximization algorithms included come from Prof. Zaiwen Wen's [lecture notes](http://faculty.bicmr.pku.edu.cn/~wenzw/bigdata/lect-submodular.pdf).