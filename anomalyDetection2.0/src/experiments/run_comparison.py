import sys
import os
# Add the root directory of your project to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from sklearn.model_selection import KFold
import numpy as np
import torch
from src.utils.evaluation import cross_validate_anomaly_detector, train_test_evaluate_model
from src.utils.tuning import tune_ocsvm_params, tune_iforest_params
import pandas as pd
# from tsai.all import ROCKET, create_rocket_features
from src.models.models import ROCKET
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from src.data.data_loading import DataLoading
from src.data.utils.preprocessing import create_rocket_features_max
import json

DATASET_NAME = "NGAFID_MC_C28"


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
        'num_kernels': [500],
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
        if num_kernels == 500:
            #save the model
            torch.save(rocket.state_dict(), f'/Users/manasdubey2022/Desktop/NGAFID/Codebase/anomalyDetection2.0/src/models/saved_rocket_models/rocket_model_{num_kernels}_{DATASET_NAME}.pth')
            print(f"ROCKET model with {num_kernels} kernels saved to /Users/manasdubey2022/Desktop/NGAFID/Codebase/anomalyDetection2.0/src/models/saved_rocket_models/rocket_model_{num_kernels}_{DATASET_NAME}.pth")


        # go through the normal and anomaly data and print the shapes of the data
        for x in normal_data:
            print(x.shape)
            # print(y.shape)
            break
        
        # Extract features
        print("\nExtracting features...")
        print("Processing normal data...")
        normal_features = create_rocket_features_max(normal_data, rocket)
        print("Processing anomaly data...")
        anomalous_features = create_rocket_features_max(anomaly_data, rocket)


        print("Features extraction completed")
        print(f"Normal features shape: {normal_features.shape}")
        print(f"Anomalous features shape: {anomalous_features.shape}")



        
        # One-Class SVM
        print("\nTraining One-Class SVM...")

        print("Running cross-validation...")
        ocsvm_results = train_test_evaluate_model(OneClassSVM, normal_features, anomalous_features, ocsvm_param_grid)
        print("One-Class SVM training completed")

        #save the json results
        with open(f'results/ocsvm_results_num_kernels_{num_kernels}_{DATASET_NAME}.json', 'w') as f:
            json.dump(ocsvm_results, f)
        
        # Isolation Forest
        print("\nTraining Isolation Forest...")

        print("Running cross-validation...")
        iforest_results = train_test_evaluate_model(IsolationForest, normal_features, anomalous_features, iforest_param_grid)
        print("Isolation Forest training completed")
        #save the json results
        with open(f'results/iforest_results_num_kernels_{num_kernels}_{DATASET_NAME}.json', 'w') as f:
            json.dump(iforest_results, f)
        
        
        # Save results for this kernel configuration
    
    return all_results

if __name__ == "__main__":
    
    # Set the environment variable for MPS fallback
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    

    # Load your data
    data_paths = ["/Users/manasdubey2022/Desktop/NGAFID/Codebase/data/NGAFID_MC_C37.csv", "/Users/manasdubey2022/Desktop/NGAFID/Codebase/data/NGAFID_MC_C28.csv"]
    for data_path in data_paths:
    # data_path = "/Users/manasdubey2022/Desktop/NGAFID_Data_Processor/NGAFID_C37_split.csv" # more data
    # data_path = "/Users/manasdubey2022/Desktop/NGAFID_Data_Processor/NGAFID_C37_split_100_events.csv" # smaller more balanced data for 100 events worth
        if data_path == "/Users/manasdubey2022/Desktop/NGAFID/Codebase/data/NGAFID_MC_C37.csv":
            DATASET_NAME = "NGAFID_MC_C37"
        elif data_path == "/Users/manasdubey2022/Desktop/NGAFID/Codebase/data/NGAFID_MC_C28.csv":
            DATASET_NAME = "NGAFID_MC_C28"
        elif data_path == "/Users/manasdubey2022/Desktop/NGAFID_Data_Processor/NGAFID_C37_split.csv":
            DATASET_NAME = "NGAFID_C37_split"
        elif data_path == "/Users/manasdubey2022/Desktop/NGAFID_Data_Processor/NGAFID_C37_split_100_events.csv":
            DATASET_NAME = "NGAFID_C37_split_100_events"
        
        data_loading = DataLoading()
        data = data_loading.load_data(data_path)
        data = data_loading.min_max_scaling(INPUT_COLUMNS, data)
        data = data.dropna()
        print(data.shape)


        folded_datasets = data_loading.get_folded_datasets('bce',data,5)

        anomalies_loader, normal_loader = data_loading.create_anomaly_normal_loaders(folded_datasets, batch_size=32)
        



        print(F"Running experiment for ROCKET {DATASET_NAME}...")
        run_experiment(normal_loader, anomalies_loader)



    # Run experiment
    # results = run_experiment(normal_loader, anomalies_loader)

    # # Print results
    # for num_kernels, results in results.items():
    #     print(f"\nResults for {num_kernels} kernels:")
    #     for model, metrics in results.items():
    #         print(f"\n{model.upper()} Results:")
    #         for metric, (mean, std) in metrics.items():
    #             print(f"{metric}: {mean:.4f} ± {std:.4f}")

    