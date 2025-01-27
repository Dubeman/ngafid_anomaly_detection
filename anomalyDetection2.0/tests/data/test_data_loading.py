import pytest
import torch
import pandas as pd
import numpy as np
from src.data.data_loading import DataLoading, BaseAnomalyDataset, INPUT_COLUMNS
from torch.utils.data import TensorDataset, DataLoader

@pytest.fixture
def ngafid_sample():
    """Create a small sample of NGAFID-like data for testing."""
    n_samples = 100
    data = {
        'id': np.repeat(range(10), 10),  # 10 flights with 10 timestamps each
        'volt1': np.random.normal(0, 1, n_samples),
        'volt2': np.random.normal(0, 1, n_samples),
        'amp1': np.random.normal(0, 1, n_samples),
        'amp2': np.random.normal(0, 1, n_samples),
        'FQtyL': np.random.normal(0, 1, n_samples),
        'FQtyR': np.random.normal(0, 1, n_samples),
        'E1 FFlow': np.random.normal(0, 1, n_samples),
        'E1 OilT': np.random.normal(0, 1, n_samples),
        'E1 OilP': np.random.normal(0, 1, n_samples),
        'E1 RPM': np.random.normal(0, 1, n_samples),
        'E1 CHT1': np.random.normal(0, 1, n_samples),
        'E1 CHT2': np.random.normal(0, 1, n_samples),
        'E1 CHT3': np.random.normal(0, 1, n_samples),
        'E1 CHT4': np.random.normal(0, 1, n_samples),
        'E1 EGT1': np.random.normal(0, 1, n_samples),
        'E1 EGT2': np.random.normal(0, 1, n_samples),
        'E1 EGT3': np.random.normal(0, 1, n_samples),
        'E1 EGT4': np.random.normal(0, 1, n_samples),
        'OAT': np.random.normal(0, 1, n_samples),
        'IAS': np.random.normal(0, 1, n_samples),
        'VSpd': np.random.normal(0, 1, n_samples),
        'NormAc': np.random.normal(0, 1, n_samples),
        'AltMSL': np.random.normal(0, 1, n_samples),
        'plane_id': np.repeat(1, n_samples),
        'split': np.repeat(range(5), 20),  # 5 splits
        'before_after': np.repeat([0, 1], 50)  # alternating pre/post maintenance
    }
    return pd.DataFrame(data)

class TestNGAFIDDataLoading:
    def test_data_separation(self, ngafid_sample):
        """Test separation of data into anomalies and normal based on before_after."""
        loader = DataLoading(ngafid_sample)
        df = loader.load_data()
        df = loader.min_max_scaling(INPUT_COLUMNS)
        
        folded_datasets = loader.get_folded_datasets('bce', df, 5)
        
        # Separate data into anomalies and normal
        anomalies = []
        normal_data = []
        for fold in folded_datasets:
            for data, label in fold:
                if label == 1:  # post maintenance
                    normal_data.append(data.permute(1, 0))
                else:  # pre maintenance
                    anomalies.append(data.permute(1, 0))
        
        # Check if data was separated correctly
        assert len(anomalies) > 0, "No anomalies found"
        assert len(normal_data) > 0, "No normal data found"
        
        # Check data shapes
        assert anomalies[0].shape[0] == 23, "Wrong feature dimension"
        assert normal_data[0].shape[0] == 23, "Wrong feature dimension"

    def test_dataloader_creation(self, ngafid_sample):
        """Test creation of DataLoaders with correct batch properties."""
        loader = DataLoading(ngafid_sample)
        df = loader.load_data()
        df = loader.min_max_scaling(INPUT_COLUMNS)
        
        folded_datasets = loader.get_folded_datasets('bce', df, 5)
        
        # Create DataLoaders
        anomalies = []
        normal_data = []
        for fold in folded_datasets:
            for data, label in fold:
                if label == 1:
                    normal_data.append(data.permute(1, 0))
                else:
                    anomalies.append(data.permute(1, 0))
        
        anomalies_dataset = torch.stack(anomalies)
        normal_dataset = torch.stack(normal_data)
        
        batch_size = 32
        anomalies_loader = DataLoader(anomalies_dataset, batch_size=batch_size, shuffle=True)
        normal_loader = DataLoader(normal_dataset, batch_size=batch_size, shuffle=True)
        
        # Test batch properties
        for batch in anomalies_loader:
            assert batch.shape[0] <= batch_size
            assert batch.shape[1] == 23
            break
            
        for batch in normal_loader:
            assert batch.shape[0] <= batch_size
            assert batch.shape[1] == 23
            break

    def test_min_max_scaling_range(self, ngafid_sample):
        """Test that min-max scaling produces values in [0,1] range."""
        loader = DataLoading(ngafid_sample)
        df = loader.load_data()
        scaled_df = loader.min_max_scaling(INPUT_COLUMNS)
        
        for col in INPUT_COLUMNS:
            assert scaled_df[col].min() >= 0, f"{col} has values below 0"
            assert scaled_df[col].max() <= 1, f"{col} has values above 1"

    def test_fold_distribution(self, ngafid_sample):
        """Test that data is distributed properly across folds."""
        loader = DataLoading(ngafid_sample)
        df = loader.load_data()
        df = loader.min_max_scaling(INPUT_COLUMNS)
        
        n_folds = 5
        folded_datasets = loader.get_folded_datasets('bce', df, n_folds)
        
        # Check number of folds
        assert len(folded_datasets) == n_folds, f"Expected {n_folds} folds"
        
        # Check that each fold has both normal and anomaly data
        for fold in folded_datasets:
            labels = [label.item() for _, label in fold]
            assert 0 in labels, "Fold missing anomaly data"
            assert 1 in labels, "Fold missing normal data"

if __name__ == '__main__':
    pytest.main([__file__])