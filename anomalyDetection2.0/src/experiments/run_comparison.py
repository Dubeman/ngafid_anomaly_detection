import sys
import os
# Add the root directory of your project to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from sklearn.model_selection import KFold
import numpy as np
import torch
from src.utils.evaluation import cross_validate_anomaly_detector
from src.utils.tuning import tune_ocsvm_params, tune_iforest_params
import pandas as pd
from tsai.all import ROCKET, create_rocket_features
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from src.data.data_loading import DataLoading
# from src.data.utils.preprocessing import create_rocket_features_max




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

def run_experiment(normal_data: torch.utils.data.DataLoader, anomaly_data: torch.utils.data.DataLoader):
    """Run complete experiment comparing different models.
    
    Args:
        normal_data: DataLoader containing normal flight data
        anomaly_data: DataLoader containing anomalous flight data
        
    Returns:
        dict: Results dictionary containing metrics for each model
    """
    # Setup parameters
    device = 'cpu'
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rocket_params = {
        'num_kernels': [100, 500, 1000, 5000],
        'kernel_sizes': [7, 9, 11],
        'channels_in': 23,
        'sequence_length': 4096
    }

        # Define parameter grids
    ocsvm_param_grid = {
        'kernel': ['rbf'],
        'nu': [0.01, 0.1, 0.5],
        'gamma': ['scale', 'auto']
    }
    
    iforest_param_grid = {
        'n_estimators': [100, 200],
        'contamination': [0.1],
        'max_samples': ['auto'],
        'random_state': [42]
    }
    
    all_results = {}
    
    # Run experiments for each kernel configuration
    for num_kernels in rocket_params['num_kernels']:
        print(f"\n{'='*50}")
        print(f"Starting experiment with {num_kernels} kernels")
        print(f"{'='*50}\n")
        
        # Initialize ROCKET model
        print("Initializing ROCKET model...")
        rocket = ROCKET(
            c_in=rocket_params['channels_in'],
            seq_len=rocket_params['sequence_length'], 
            n_kernels=num_kernels,
            kss=rocket_params['kernel_sizes']
        ).to(device)
        print("ROCKET model initialized successfully")
        
        # Extract features
        print("\nExtracting features...")
        print("Processing normal data...")
        normal_features = create_rocket_features(normal_data, rocket)
        print("Processing anomaly data...")
        anomalous_features = create_rocket_features(anomaly_data, rocket)
        print(f"Features extracted successfully. Shape: {normal_features.shape}")

        # Run models
        results = {}
        
        # One-Class SVM
        print("\nTraining One-Class SVM...")

        print("Running cross-validation...")
        _, ocsvm_avg = cross_validate_anomaly_detector(
            OneClassSVM, 
            normal_features, 
            anomalous_features, 
            kf,
            ocsvm_param_grid
        )
        results['ocsvm'] = ocsvm_avg
        print("One-Class SVM training completed")
        
        # Isolation Forest
        print("\nTraining Isolation Forest...")

        print("Running cross-validation...")
        _, iforest_avg = cross_validate_anomaly_detector(
            IsolationForest, 
            normal_features, 
            anomalous_features, 
            kf,
            iforest_param_grid
        )
        results['iforest'] = iforest_avg
        print("Isolation Forest training completed")
        
        # Save results for this kernel configuration
        print("\nSaving results...")
        results_df = pd.DataFrame(results)
        results_df.to_csv(f'results/comparison_results_num_kernels_{num_kernels}.csv')
        print(f"Results saved to: results/comparison_results_num_kernels_{num_kernels}.csv")
        
        all_results[num_kernels] = results
        print(f"\nCompleted experiment with {num_kernels} kernels")
    
    return all_results

if __name__ == "__main__":
    
    # Set the environment variable for MPS fallback
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    
    # Load your data
    data_path = "/Users/manasdubey2022/Desktop/NGAFID/Codebase/data/NGAFID_MC_C37.csv"
    data_loading = DataLoading(data_path)
    data = data_loading.load_data()
    data = data_loading.min_max_scaling(INPUT_COLUMNS)
    print(data.head(5))


    folded_datasets = data_loading.get_folded_datasets('bce',data,5)

    anomalies_loader, normal_loader = data_loading.create_anomaly_normal_loaders(folded_datasets, batch_size=32)
    

    #print the size of the loaders
    # print(len(anomalies_loader))
    # print(len(normal_loader))



    # Run experiment
    results = run_experiment(normal_loader, anomalies_loader)

    # Print results
    for num_kernels, results in results.items():
        print(f"\nResults for {num_kernels} kernels:")
        for model, metrics in results.items():
            print(f"\n{model.upper()} Results:")
            for metric, (mean, std) in metrics.items():
                print(f"{metric}: {mean:.4f} ± {std:.4f}")

    