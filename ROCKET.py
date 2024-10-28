import os
import torch
import torch.nn as nn
import numpy as np
from fastprogress import progress_bar
from utils.ChunkedCSVLoader import ChunkedCSVDataset
from torch.utils.data import DataLoader

# Set the environment variable for MPS fallback
# os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

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

def create_rocket_features(dl, model, save_file=None, verbose=False):
    _x_out, _y_out = [], []
    for i, (xb, yb) in enumerate(progress_bar(dl, display=verbose, leave=False)):
        try:
            # Ensure the input tensor is of the correct type and device
            xb = xb.clone().detach().float().to(model.device)
            yb = yb.clone().detach().float().to(model.device)
            features = model(xb).cpu()
            _x_out.append(features)
            _y_out.append(yb.cpu())
            
            if save_file and (i + 1) % 5 == 0: # Save every 5 batches
                # Concatenate features and labels
                features_and_labels = torch.cat([torch.cat(_x_out), torch.cat(_y_out).unsqueeze(1)], dim=1)
                np.savetxt(save_file, features_and_labels.numpy(), delimiter=",")

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

def add_channel_dimension(num_channels):
    '''
    Transforms a 2D tensor to a 3D tensor by adding a channel dimension.
    '''
    def transform(X):
        X = torch.tensor(X, dtype=torch.float32)
        return X.unsqueeze(0) if num_channels == 1 else X.view(num_channels, -1)
    return transform


# Example usage
if __name__ == "__main__":
    # Create a ChunkedCSVDataset which reads the data in chunks and pass it to a DataLoader for batching.
    dataset = ChunkedCSVDataset('/shared/rc/gamts/Codebase/data/NGAFID_train_data.csv', 'before_after', chunk_size=10000, transform=add_channel_dimension(1))
    dataloader = DataLoader(dataset, batch_size=512, shuffle=False)

    # Take one batch from the DataLoader
    for batch in dataloader:
        break

    # Display the type and shape of the batch
    print(f"Batch type: {type(batch)}")  # Expect torch.Tensor
    print(f"Batch element types: {type(batch[0])}, {type(batch[1])}")
    print(f"Batch shapes: {batch[0].size()}, {batch[1].size()}")

    # Extract sequence length from the batch
    seq_len = batch[0].size()[-1]  # Sequence length = number of features

    # Determine the device to use (GPU if available, otherwise CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize the ROCKET model
    model = ROCKET(c_in=1, seq_len=seq_len, n_kernels=10, kss=[3, 5, 7], device=device, verbose=True)
    model.to(device)  # Ensure the model is on the correct device

    # Pass the batch through the model and display the output shape
    output = model(batch[0].to(device))  # Ensure the input tensor is on the correct device
    print(f"Output shape: {output.shape}")