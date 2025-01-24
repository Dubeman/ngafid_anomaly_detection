import pytest
import torch
import pandas as pd
import numpy as np
from src.data.data_loading import DataLoading, BaseAnomalyDataset, INPUT_COLUMNS
from torch.utils.data import TensorDataset, DataLoader


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    # Create a temporary CSV file with test data
    data = {
        'id': range(100),
        'volt1': np.random.normal(0, 1, 100),
        'volt2': np.random.normal(0, 1, 100),
        'amp1': np.random.normal(0, 1, 100),
        'amp2': np.random.normal(0, 1, 100),
        'FQtyL': np.random.normal(0, 1, 100),
        'FQtyR': np.random.normal(0, 1, 100),
        'E1 FFlow': np.random.normal(0, 1, 100),
        'E1 OilT': np.random.normal(0, 1, 100),
        'E1 OilP': np.random.normal(0, 1, 100),
        'E1 RPM': np.random.normal(0, 1, 100),
        'E1 CHT1': np.random.normal(0, 1, 100),
        'E1 CHT2': np.random.normal(0, 1, 100),
        'E1 CHT3': np.random.normal(0, 1, 100),
        'E1 CHT4': np.random.normal(0, 1, 100),
        'E1 EGT1': np.random.normal(0, 1, 100),
        'E1 EGT2': np.random.normal(0, 1, 100),
        'E1 EGT3': np.random.normal(0, 1, 100),
        'E1 EGT4': np.random.normal(0, 1, 100),
        'OAT': np.random.normal(0, 1, 100),
        'IAS': np.random.normal(0, 1, 100),
        'VSpd': np.random.normal(0, 1, 100),
        'NormAc': np.random.normal(0, 1, 100),
        'AltMSL': np.random.normal(0, 1, 100),
    }
    df = pd.DataFrame(data)
    test_file = 'test_data.csv'
    df.to_csv(test_file, index=False)
    return test_file

class TestBaseAnomalyDataset:
    def test_init(self):
        """Test dataset initialization."""
        data = torch.randn(100, 23)  # 100 samples, 23 features
        labels = torch.randint(0, 2, (100,))  # binary labels
        dataset = BaseAnomalyDataset(data, labels)
        
        assert len(dataset) == 100
        assert torch.equal(dataset.data, data)
        assert torch.equal(dataset.labels, labels)

    def test_getitem_with_labels(self):
        """Test __getitem__ with labels."""
        data = torch.randn(100, 23)
        labels = torch.randint(0, 2, (100,))
        dataset = BaseAnomalyDataset(data, labels)
        
        item_data, item_label = dataset[0]
        assert torch.equal(item_data, data[0])
        assert torch.equal(item_label, labels[0])

    def test_getitem_without_labels(self):
        """Test __getitem__ without labels."""
        data = torch.randn(100, 23)
        dataset = BaseAnomalyDataset(data)
        
        item_data = dataset[0]
        assert torch.equal(item_data, data[0])

class TestDataLoading:
    def test_load_data(self, sample_data):
        """Test data loading functionality."""
        loader = DataLoading(sample_data)
        df = loader.load_data()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 100
        assert all(col in df.columns for col in INPUT_COLUMNS)

    def test_min_max_scaling(self, sample_data):
        """Test min-max scaling functionality."""
        loader = DataLoading(sample_data)
        df = loader.load_data()
        scaled_df = loader.min_max_scaling(INPUT_COLUMNS)
        
        # Check if values are scaled between 0 and 1
        for col in INPUT_COLUMNS:
            assert scaled_df[col].min() >= 0
            assert scaled_df[col].max() <= 1

    def test_create_anomaly_normal_loaders(self, sample_data):
        """Test creation of anomaly and normal data loaders."""
        # Create sample folded datasets
        data = torch.randn(10, 23, 100)  # 10 samples, 23 features, 100 timesteps
        labels = torch.randint(0, 2, (10,))
        folded_datasets = [TensorDataset(data[i], labels[i:i+1]) for i in range(10)]
        
        loader = DataLoading(sample_data)
        anomalies_loader, normal_loader = loader.create_anomaly_normal_loaders(
            folded_datasets, batch_size=2
        )
        
        assert isinstance(anomalies_loader, DataLoader)
        assert isinstance(normal_loader, DataLoader)

if __name__ == '__main__':
    pytest.main([__file__]) 