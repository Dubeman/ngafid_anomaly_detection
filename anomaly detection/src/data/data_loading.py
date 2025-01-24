import pandas as pd
import numpy as np
from sklearn import preprocessing 
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from typing import List, Tuple, Optional, Union
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from utils.preprocessing import get_dataset, prepare_for_training

# Constants
INPUT_COLUMNS = [
    'volt1', 'volt2', 'amp1', 'amp2', 'FQtyL', 'FQtyR',
    'E1 FFlow', 'E1 OilT', 'E1 OilP', 'E1 RPM',
    'E1 CHT1', 'E1 CHT2', 'E1 CHT3', 'E1 CHT4',
    'E1 EGT1', 'E1 EGT2', 'E1 EGT3', 'E1 EGT4',
    'OAT', 'IAS', 'VSpd', 'NormAc', 'AltMSL'
]

class BaseAnomalyDataset(Dataset):
    """Base class for anomaly detection datasets."""
    
    def __init__(self, data: torch.Tensor, labels: Optional[torch.Tensor] = None):
        """
        Initialize the dataset.
        
        Args:
            data (torch.Tensor): Input data tensor
            labels (torch.Tensor, optional): Labels tensor if available
        """
        self.data = data
        self.labels = labels

    def __len__(self) -> int:
        """Return the total number of samples."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: 
                Either just the data tensor or a tuple of (data, label)
        """
        if self.labels is not None:
            return self.data[idx], self.labels[idx]
        return self.data[idx]

class DataLoading:
    """Class for loading and preprocessing flight data for anomaly detection."""
    
    def __init__(self, data_path: str):
        """
        Initialize the DataLoading class.
        
        Args:
            data_path (str): Path to the data file
        """
        self.data_path = data_path
        self.df: Optional[pd.DataFrame] = None

    def load_data(self) -> pd.DataFrame:
        """
        Load data from CSV file with optimized dtypes.
        
        Returns:
            pd.DataFrame: Loaded and preprocessed DataFrame
        """
        df_test = pd.read_csv(self.data_path, nrows=100)
        float_cols = [c for c in df_test if df_test[c].dtype == "float64"]
        float32_cols = {c: np.float32 for c in float_cols}

        df = pd.read_csv(self.data_path, engine='c', dtype=float32_cols)
        df['id'] = df.id.astype('int32')
        self.df = df.dropna()
        print(df.head(5))
        return df
    
    def min_max_scaling(self, input_columns: List[str], df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Apply Min-Max scaling to specified columns.
        
        Args:
            input_columns (List[str]): List of column names to scale
            df (Optional[pd.DataFrame]): DataFrame to scale, uses self.df if None
            
        Returns:
            pd.DataFrame: Scaled DataFrame
            
        Raises:
            ValueError: If string values are found in numeric columns
        """
        df = self.df if df is None else df
        qt = preprocessing.MinMaxScaler()
        try:
            qt.fit(df.loc[:, input_columns].sample(100000, random_state=0))
        except ValueError as e:
            if "could not convert string to float" in str(e):
                problematic_string = str(e).split(": ")[1].strip("'")
                print(f"Error: Could not convert string to float: '{problematic_string}'")
                for col in input_columns:
                    if problematic_string in df[col].astype(str).values:
                        print(f"Found problematic string in column: '{col}'")
            raise

        arr = df.loc[:, input_columns].values
        res = qt.transform(arr)

        for i, col in tqdm(enumerate(input_columns)):
            df.loc[:, col] = res[:, i]

        return df
    
    def get_folded_datasets(self, MODEL_LOSS_TYPE: str, df: pd.DataFrame, NFOLD: int) -> List[Union[TensorDataset, Tuple[TensorDataset, TensorDataset]]]:
        """
        Create folded datasets for cross-validation.
        
        Args:
            MODEL_LOSS_TYPE (str): Type of loss function ('bce' or 'mse')
            df (pd.DataFrame): Input DataFrame
            NFOLD (int): Number of folds
            
        Returns:
            List[Union[TensorDataset, Tuple[TensorDataset, TensorDataset]]]: List of folded datasets
        """
        folded_datasets = []

        for i in range(NFOLD):
            if MODEL_LOSS_TYPE == 'bce':
                folded_datasets.append(get_dataset(df[df.split == i]))
            elif MODEL_LOSS_TYPE == 'mse':
                after = get_dataset(df[(df.split == i) & (df.before_after == 1)])
                before = get_dataset(df[(df.split == i) & (df.before_after == 0)])
                folded_datasets.append((after, before))

        return folded_datasets
    

    def get_train_and_val_for_fold(self,folded_datasets,fold,MODEL_LOSS_TYPE='bce',NFOLD=5,AUGMENT=None, PREDICT=False): 
        
        if MODEL_LOSS_TYPE == 'bce':
            train = []
            for i in range(NFOLD):
                if i == fold:
                    val_ds = folded_datasets[i]
                else:
                    train.append(folded_datasets[i])
        elif MODEL_LOSS_TYPE == 'mse':
            train = []
            for i in range(NFOLD):
                if i == fold:
                    val_ds = folded_datasets[i][0].concatenate(folded_datasets[i][1])
                else:
                    train.append(folded_datasets[i][0])

        train_ds = None
        for ds in train:
            if isinstance(ds, torch.utils.data.TensorDataset):
                ds_tensors = ds.tensors
                for tensor in ds_tensors:
                    train_ds = tensor if train_ds is None else torch.cat((train_ds, tensor))
            else:
                train_ds = ds if train_ds is None else torch.cat((train_ds, ds))

        mse_val_ds = None if not MODEL_LOSS_TYPE == 'mse' else prepare_for_training(val_ds, shuffle=False,  predict=True)
        train_ds = prepare_for_training(train_ds, shuffle=True, repeat = True, predict=PREDICT, aug = AUGMENT)
        val_ds = prepare_for_training(val_ds, shuffle=False,  predict=PREDICT)

        return train_ds, val_ds, mse_val_ds
    
    def create_anomaly_normal_loaders(self, folded_datasets: List[TensorDataset], batch_size: int = 32) -> Tuple[DataLoader, DataLoader]:
        """
        Creates separate DataLoaders for anomaly and normal data.
        
        Args:
            folded_datasets (List[TensorDataset]): List of folded datasets
            batch_size (int): Batch size for the DataLoaders
            
        Returns:
            Tuple[DataLoader, DataLoader]: Tuple containing (anomalies_loader, normal_loader)
        """
        anomalies = []  # post maintenance (0)
        normal_data = []  # pre maintenance (1)
        
        for fold in folded_datasets:
            for data, label in fold:
                if label == 1:
                    normal_data.append(data.permute(1, 0))
                else:
                    anomalies.append(data.permute(1, 0))

        anomalies_dataset = torch.stack(anomalies)
        normal_dataset = torch.stack(normal_data)

        anomalies_loader = DataLoader(anomalies_dataset, batch_size=batch_size, shuffle=True)
        normal_loader = DataLoader(normal_dataset, batch_size=batch_size, shuffle=True)
        
        return anomalies_loader, normal_loader
    
    