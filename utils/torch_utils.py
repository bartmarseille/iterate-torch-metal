import torch
import numpy as np

TORCH_DTYPES = { 'float32': torch.float32, 'float64': torch.float64}


class MpsDataset(torch.utils.data.Dataset):
    
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = X
        self.Y = Y
    
    def __getitem__(self, idx):
        return torch.tensor([self.X[idx]], dtype=torch.float32), torch.tensor([self.Y[idx]], dtype=torch.float32)
        
    def __len__(self): return len(self.Y)


def get_device():
    device = 'cpu'
    if torch.backends.mps.is_available():
        # this ensures that the current MacOS version is at least 12.3+
        device=torch.device("mps")
        # # this ensures that the current current PyTorch installation was built with MPS activated.
        # print(f'torch mps backend built: {torch.backends.mps.is_built()}')
    elif torch.backends.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print('pytorch using device:', device)
    return device


def reset_seed(seed):
    torch.manual_seed(seed)
    # torch.backends.mps.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.mps.deterministic = True
    torch.backends.cudnn.benchmark = False

    np.random.seed(seed)

