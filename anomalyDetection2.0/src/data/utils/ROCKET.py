import os
import torch
import torch.nn as nn
import numpy as np
from fastprogress import progress_bar
import pandas as pd


# Set the environment variable for MPS fallback
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# Verify NumPy version
print("NumPy version:", np.__version__)

# Verify PyTorch version
print("PyTorch version:", torch.__version__)

class ROCKET(nn.Module):
    def __init__(self, c_in, seq_len, n_kernels, kss, device=None, verbose=False):
        '''
        Input: is a 3d torch tensor of type torch.float32. When used with univariate TS,
        make sure you transform the 2d to 3d by adding unsqueeze(1).
        c_in: number of channels or features. For univariate c_in is 1.
        seq_len: sequence length
        '''
        super().__init__()
        self.device = device or torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        kss = [ks for ks in kss if ks < seq_len]
        convs = nn.ModuleList()
        for i in range(n_kernels):
            ks = np.random.choice(kss)
            dilation = 2**np.random.uniform(0, np.log2((seq_len - 1) // (ks - 1)))
            padding = int((ks - 1) * dilation // 2) if np.random.randint(2) == 1 else 0
            weight = torch.randn(1, c_in, ks)
            weight -= weight.mean()
            bias = 2 * (torch.rand(1) - .5)
            layer = nn.Conv1d(c_in, 1, ks, padding=2 * padding, dilation=int(dilation), bias=True)
            layer.weight = torch.nn.Parameter(weight, requires_grad=False)
            layer.bias = torch.nn.Parameter(bias, requires_grad=False)
            convs.append(layer)
        self.convs = convs
        self.n_kernels = n_kernels
        self.kss = kss
        self.to(device=self.device)
        self.verbose = verbose

    def forward(self, x):
        _output = []
        for i in progress_bar(range(self.n_kernels), display=self.verbose, leave=False):
            out = self.convs[i](x)
            _max = out.max(dim=-1)[0]
            _ppv = torch.gt(out, 0).sum(dim=-1).float() / out.shape[-1]
            _output.append(_max)
            _output.append(_ppv)
        return torch.cat(_output, dim=1)

¸
        except Exception as e:
            print(f"Error processing batch {i}")
            print(f"Input batch shape: {xb.shape}")
            print(f"Target batch shape: {yb.shape}")
            print(f"Input tensor type: {type(xb)}, dtype: {xb.dtype}")
            print(f"Target tensor type: {type(yb)}, dtype: {yb.dtype}")
            print(f"Conv layer type: {type(model.convs[i])}")
            print(f"Conv layer weights dtype: {model.convs[i].weight.dtype}")
            if model.convs[i].bias is not None:
                print(f"Conv layer bias dtype: {model.convs[i].bias.dtype}")
            print(f"Error message: {e}")
            raise e

# Example usage
if __name__ == "__main__":
#    Example input tensor
    # bs = 16
    # c_in = 7  # aka channels, features, variables, dimensions
    # c_out = 2
    # seq_len = 15
    # xb = torch.randn(bs, c_in, seq_len)
    # for i in range(3):
    #     print("Example input tensor:")
    #     print(xb[i])

    # print(xb.shape)

    # m = ROCKET(c_in, seq_len, n_kernels=1_000, kss=[7, 9, 11])
    # out = m(xb)
    # print(out.shape)
    # print(out.dtype)


    # file_path = '/Users/manasdubey2022/Desktop/NGAFID/Codebase/data/planes/cleaned_flights/34_cleaned.csv'






