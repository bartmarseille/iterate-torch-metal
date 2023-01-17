import torch
import numpy as np
import random
import os
from packaging.version import parse, Version


class MpsDataset(torch.utils.data.Dataset):
    
    def __init__(self, X: np.ndarray, Y: np.ndarray, device: torch.device):
        self.X = X
        self.Y = Y
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        if not self.device == torch.device('mps'):
            print(f'overruling MPS device, using {self.device}')
    
    def __getitem__(self, idx):
        if self.device == torch.device('mps'):
            # cast data type to float32
            return torch.tensor(self.X[idx,:], dtype=torch.float32), torch.tensor(self.Y[idx,:], dtype=torch.float32)
        else:
            return torch.tensor(self.X[idx,:]), torch.tensor(self.Y[idx,:])
        
    def __len__(self): return len(self.Y)

    def get_data(self):
        if self.device == torch.device('mps'):
            # cast data type to float32
            return torch.tensor(self.X, dtype=torch.float32).to(device=self.device), torch.tensor(self.Y, dtype=torch.float32).to(device=self.device)
        else:
            return torch.tensor(self.X), torch.tensor(self.Y)



def get_device():        
    if not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was not built with MPS enabled.")

    if torch.backends.mps.is_available():
        device=torch.device("mps")
    elif torch.backends.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f'pytorch using device: {device}')
    return device


def reset_seed(seed: int = 33) -> None:
    """
    Sets random seed into PyTorch, TensorFlow, Numpy and Random.
    Args:
        seed: random seed
    """
    try:
        import torch
    except ImportError:
        pass
    else:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.mps.deterministic = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    try:
        import tensorflow as tf
    except ImportError:
        pass
    else:
        if parse(tf.__version__) >= Version("2.0.0"):
            tf.random.set_seed(seed)
        elif parse(tf.__version__) <= Version("1.13.2"):
            tf.set_random_seed(seed)
        else:
            tf.compat.v1.set_random_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

# # For Pytorch
# def reset_seed(seed: int = 42):
#     np.random.seed(seed)
#     random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     # When running on the MPS backend (Apple M1 GPU)
#     torch.backends.mps.deterministic = True
#     # When running on the CuDNN backend, two further options must be set
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#     # Set a fixed value for the hash seed
#     os.environ["PYTHONHASHSEED"] = str(seed)
#     print(f"Random seed set as {seed}")

# # For TensorFlow
# def set_seed(seed: int = 42) -> None:
#   random.seed(seed)
#   np.random.seed(seed)
#   tf.random.set_seed(seed)
#   tf.experimental.numpy.random.seed(seed)
#   tf.set_random_seed(seed)
#   # When running on the CuDNN backend, two further options must be set
#   os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
#   os.environ['TF_DETERMINISTIC_OPS'] = '1'
#   # Set a fixed value for the hash seed
#   os.environ["PYTHONHASHSEED"] = str(seed)
#   print(f"Random seed set as {seed}")


