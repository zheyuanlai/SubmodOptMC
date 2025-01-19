# Submodular optimization of the KL

We use submodular minimization on the KL divergence between original transition probability matrix P and the outer product of keep-S-in and leave-S-out matrices, subject to a cardinality constraint.

To test the main logic, run the `main.py` script:
```
python main.py
``` 

The script `main.py` also includes testing part, which compares the KL divergence between optimal `S` and other non-optimal `S`'s.

Since the optimization objective is subject to a knapsack constraint, we apply greedy algorithm instead of Lovasz extension.