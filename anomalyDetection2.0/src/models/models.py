import os
import torch
import torch.nn as nn
import numpy as np
from fastprogress import progress_bar
import pandas as pd
# from utils.preprocessing import preprocess_data, pad_group_constant

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
            # print("Max shape:", _max.shape)
            # print("Max:", _max)
            _ppv = torch.gt(out, 0).sum(dim=-1).float() / out.shape[-1]
            _output.append(_max)
            _output.append(_ppv)
        return torch.cat(_output, dim=1)
    

    def print_module_list(self):
        """
        Print the details of the nn.ModuleList (self.convs).
        """
        print(f"Number of kernels: {len(self.convs)}")
        for i, conv in enumerate(self.convs):
            print(f"Kernel {i + 1}:")
            print(f"  Kernel Size: {conv.kernel_size}")
            print(f"  Padding: {conv.padding}")
            print(f"  Dilation: {conv.dilation}")
            print(f"  Weight Shape: {conv.weight.shape}")
            print(f"  Bias: {conv.bias.detach().cpu().numpy() if conv.bias is not None else None}")
            #print the layer dimensions
            print(f"  Layer Dimensions: {conv.weight.shape[0]} x {conv.weight.shape[1]} x {conv.weight.shape[2]}")
            print("-" * 30)

    

    def get_kernel_responses(self, x):
        """
        Get the responses of each convolutional kernel for the input.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, c_in, seq_len).
        
        Returns:
            List[torch.Tensor]: List of kernel responses for each convolutional layer.
        """
        kernel_responses = []
        for conv in self.convs:
            response = conv(x)  # Apply the convolution
            kernel_responses.append(response)
        return kernel_responses


def main():
        # Example usage
    model = ROCKET(c_in=23, seq_len=4096, n_kernels=100, kss=[7, 9, 11])
    print(model)


if __name__ == "__main__":
    main()





