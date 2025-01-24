# Submodular optimization of KL divergence

We use submodular minimization on the KL divergence between original transition probability matrix P and the outer product of keep-S-in and leave-S-out matrices, subject to a cardinality constraint.

To test the main logic, run the `main.py` script:
```
python main.py
``` 

Since the optimization objective is subject to a knapsack constraint, we apply greedy algorithm instead of Lovasz extension.