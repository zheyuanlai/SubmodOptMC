# Supermodular optimization of "distance to independence"

We use supermodular minimization on the distance to independence of a subset Markov chain, subject to a cardinality constraint.

To test the main logic, run the `main.py` script:
```
python main.py
``` 

Minimizing a supermodular function $f$ is the same as maximizing a submodular function $-f$. Hence, we use the greedy algorithm for submodular maximization.