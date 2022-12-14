import numpy as np
from typing import Callable


def iterate(func: Callable, V: np.array, n: int = 0, **kwargs) -> np.array:

    if not isinstance(V, np.ndarray):
        return iterate(func, np.array(V), n, **kwargs)
    else:    
        trajectory = [V]
        for t in range(1,n):
            trajectory.append(func(trajectory[-1], **kwargs))
        return np.array(trajectory)


def logistic_map(V: np.ndarray, r=3.0) -> np.ndarray:
    """The implementation of a relative population system with:
    -  input variable vector `V`,
    -  r: reproduction parameter
    that maps to the output variable vector `V_dot`
    """
    
    x = V    # relative population size
    
    return r * x * (1 - x)


def gauss_map(V: np.ndarray, alpha=6.2, beta=-0.5) -> np.ndarray:
    """Named after Johann Carl Friedrich Gauss, the function maps the bell shaped Gaussian function similar to the logistic map.
    -  input variable vector V,
    -  parameter alpha
    -  parmeter beta
    that maps to the output variable vector `V_dot`
    """
    
    x = V

    return np.exp(-alpha * x**2) + beta
    