import numpy as np
from typing import Callable

def logistic_map(P: dict, data: np.ndarray) -> np.ndarray:
    """The implementation of a relative population system with:
    -  parameters `P`: ['r'] and,
    -  variables `data`,
    that maps to the output `data_hat` of the same type as variable `data`
    """
    r = P['r']  # reproduction rate
    x = data    # relative population size

    return r * x * (1 - x)


def gauss_map(P: dict, data: np.ndarray) -> np.ndarray:
    """Named after Johann Carl Friedrich Gauss, the function maps the bell shaped Gaussian function similar to the logistic map.
    -  parameters `P`: ['alpha', 'beta'] and,
    -  variables `data`,
    that maps to the output `data_hat` of the same type as variable `data`
    """
    alpha = P['alpha']  # 
    beta = P['beta']  # 
    x = data    # 

    return np.exp(-alpha * x**2) + beta
    

def iterate(map: Callable, P: dict, data: np.array, n: int = 0) -> np.array:

    if not isinstance(data, np.ndarray):
        return iterate(map, P, np.array(data), n)
    else:    
        new_shape = (n,) + data.shape
        trajectory = np.empty(shape=new_shape, dtype=data.dtype)

        trajectory[0] = data
        for i in range(1,n):
            trajectory[i] = map(P, trajectory[i-1])
        return trajectory