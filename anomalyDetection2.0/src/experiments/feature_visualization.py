import os
import sys
import numpy as np
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.data.data_loading import DataLoading
from src.models.models import ROCKET
import matplotlib.pyplot as plt
import torch

INPUT_COLUMNS = INPUT_COLUMNS = ['volt1',
 'volt2',
 'amp1',
 'amp2',
 'FQtyL',
 'FQtyR',
 'E1 FFlow',
 'E1 OilT',
 'E1 OilP',
 'E1 RPM',
 'E1 CHT1',
 'E1 CHT2',
 'E1 CHT3',
 'E1 CHT4',
 'E1 EGT1',
 'E1 EGT2',
 'E1 EGT3',
 'E1 EGT4',
 'OAT',
 'IAS',
 'VSpd',
 'NormAc',
 'AltMSL']

def plot_kernel_responses(kernel_responses, input_idx=4, num_kernels_to_plot=5):
    """
    Plot the kernel responses for a given input.
    
    Args:
        kernel_responses (List[torch.Tensor]): List of kernel responses from the ROCKET model.
        input_idx (int): Index of the input sample to plot responses for.
        num_kernels_to_plot (int): Number of kernels to plot.
    """
    num_kernels = len(kernel_responses)
    num_kernels_to_plot = min(num_kernels, num_kernels_to_plot)
    
    plt.figure(figsize=(15, 5 * num_kernels_to_plot))
    for i in range(num_kernels_to_plot):
        response = kernel_responses[i][input_idx].detach().cpu().numpy()
        plt.subplot(num_kernels_to_plot, 1, i + 1)
        plt.plot(response[0])  # Plot the response for the first channel
        plt.title(f"Kernel {i + 1} Response")
        plt.xlabel("Time Steps")
        plt.ylabel("Response")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    
    # Set the environment variable for MPS fallback
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    
    # Load your data
    data_path = "/Users/manasdubey2022/Desktop/NGAFID/Codebase/data/NGAFID_MC_C37.csv"
    data_loading = DataLoading()
    data = data_loading.load_data(data_path)
    data = data_loading.min_max_scaling(INPUT_COLUMNS)
    print(data.head(5))


    folded_datasets = data_loading.get_folded_datasets('bce',data,5)

    anomalies_loader, normal_loader = data_loading.create_anomaly_normal_loaders(folded_datasets, batch_size=32)

    # ROCKET params for NGAFID
    rocket_params = {
        'num_kernels': [100],
        'kernel_sizes': [7, 9, 11],
        'channels_in': 23,
        'sequence_length': 4096
    }
    # Define the device (CPU or GPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    for num_kernels in rocket_params['num_kernels']:
        # Initialize ROCKET model
        rocket = ROCKET(
            c_in=rocket_params['channels_in'],
            seq_len=rocket_params['sequence_length'], 
            n_kernels=num_kernels,
            kss=rocket_params['kernel_sizes']
        ).to(device)
        rocket.print_module_list()
        #take one tensor from normal_loader 
        normal_tensor = next(iter(normal_loader))[0]

        #print the shape of the tensor
        print(f"Shape of normal tensor: {normal_tensor.shape}")
        kernel_responses = rocket.get_kernel_responses(normal_tensor)
        print(f"Kernel responses shape: {len(kernel_responses)}")
        plot_kernel_responses(kernel_responses, input_idx=0, num_kernels_to_plot=5)

    