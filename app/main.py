def f(x):
    """Linear: f(x) = 2x"""
    return 2 * x


import numpy as np

# Functions work on arrays too — element-wise
x = np.array([0, 1, 2, 3])
print(f(x))  # [0 2 4 6]
print(g(x))  # [0 1 4 9]