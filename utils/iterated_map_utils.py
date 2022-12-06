import numpy as np
from typing import Callable


def iterate(func: Callable, data: np.array, n: int = 0, **kwargs) -> np.array:

    if not isinstance(data, np.ndarray):
        return iterate(func, np.array(data), n, **kwargs)
    else:    
        trajectory = [data]
        for t in range(1,n):
            trajectory.append(func(trajectory[-1], **kwargs))
        return np.array(trajectory)


def logistic_map(data: np.ndarray, r=3.0) -> np.ndarray:
    """The implementation of a relative population system with:
    -  variables `data`,
    -  r: reproduction parameter
    that maps to the output `data_hat` of the same type as variable `data`
    """
    
    x = data    # relative population size
    
    return r * x * (1 - x)


def gauss_map(data: np.ndarray, alpha=6.2, beta=-0.5) -> np.ndarray:
    """Named after Johann Carl Friedrich Gauss, the function maps the bell shaped Gaussian function similar to the logistic map.
    -  variables `data`,
    -  parameter alpha
    -  parmeter beta
    that maps to the output `data_hat` of the same type as variable `data`
    """
    
    x = data

    return np.exp(-alpha * x**2) + beta
    