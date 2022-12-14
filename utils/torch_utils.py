import torch
import numpy as np


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

def to_device(batch, device="cpu"):
    "Move tensors to device"
    if isinstance(batch, torch.Tensor):
        batch.to(device)
    elif isinstance(batch, dict):
        for k,v in batch.items():
            batch[k] = v.to(device)
    else:
        raise Exception(f"Can't put your batch of type {type(batch)} into device: {device}")
    return batch

def reset_seed(seed):
    torch.manual_seed(seed)
    # torch.backends.mps.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.mps.deterministic = True
    torch.backends.cudnn.benchmark = False

    np.random.seed(seed)

