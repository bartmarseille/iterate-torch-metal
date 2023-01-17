"""
Defines a test on the lorentz ODE based dataset.
"""

import numpy as np
import torch
import torch.nn as nn

from types import SimpleNamespace
import sys
sys.path.append('..')

import utils.ode_utils as ode_utils



def test(model: nn.Module, parameters: dict, device: str):
    P = parameters
    x0 = [1.5, 0.6, 0.7]

    X = ode_utils.iterate(ode_utils.lorentz_ode, x0, n=3501, **P)
    Y_dot = X[1:,:]
    X = X[:-1,:]

    X_torch = torch.tensor(X, dtype=torch.float32).to(device)
    Y_hat = model(X_torch)
    Y_hat = Y_hat.cpu().detach().numpy()

    rmse_loss = np.sqrt(np.mean((Y_hat-Y_dot)**2))

    return rmse_loss