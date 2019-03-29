# `hte` has been separated into two parts: 

1. [`forestry`](https://github.com/soerenkuenzel/forestry) is an R package that contains several algorithms including the CART algorithm, the random forests algorithm, and the gradient boosting algorithm.  
   * it has a particular focus on statistical inference and interpretability. 
   * the backend is implemented in C++ and optimized for speed. 
2. [`causalToolbox`](https://github.com/soerenkuenzel/causalToolbox) implements several CATE estimators and tools for heterogeneous treatment effect estimation. It is based on [`forestry`](https://github.com/soerenkuenzel/forestry).
