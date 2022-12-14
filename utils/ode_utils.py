import numpy as np
from typing import Callable


def iterate(func: Callable, V: np.array, n: float=1.0, dt=0.1, **kwargs) -> np.array:

    if not isinstance(V, np.ndarray):
        return iterate(func, np.array(V), n, dt, **kwargs)
    else:
        if n==1.0:
            time = np.array([1.0+dt])
        else:
            time = np.arange(1.0+dt, 1.0+(n*dt), dt)

        trajectory = [V]
        for t in time:
            trajectory.append(rk4(func, trajectory[-1], t, dt, **kwargs))
        return np.array(trajectory)


def rk4(func, V, t, dt=0.01, **kwargs):
    """
    single-step fourth-order numerical integration (RK4) method
    func: system of first order ODEs
    
    V: current state vector [y1, y2, y3, ...]
    t: current time step
    dt: discrete time step size
    **kwargs: additional parameters for ODE system
    returns: y evaluated at time t+dt
    """
   
    # evaluate derivative at several stages within time interval
    f1 = func(V, t, **kwargs)
    f2 = func(V + (f1 * (dt / 2)), t + dt / 2, **kwargs)
    f3 = func(V + (f2 * (dt / 2)), t + dt / 2, **kwargs)
    f4 = func(V + (f3 * dt), t + dt, **kwargs)

    # return an average of the derivative over t, t + dt
    return V + (dt / 6) * (f1 + (2 * f2) + (2 * f3) + f4)


def lorentz_ode(V: np.ndarray, t=1, sigma=10., beta=8./3., rho=28.) -> np.ndarray:
    '''
    V: current state vector [y1, y2, y3, ...]
    sigma: constant related to Prandtl number
    beta: geometric physical property of fluid layer
    rho: constant related to the Rayleigh number
    '''

    x, y, z = tuple(V.T)

    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = (x * y) - (beta * z)

    return np.array([dx_dt, dy_dt, dz_dt]).T
