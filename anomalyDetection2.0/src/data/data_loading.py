import pandas as pd
import numpy as np
from sklearn import preprocessing 
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from typing import List, Tuple, Optional, Union
import torch
import json

from .utils.preprocessing import get_dataset, prepare_for_training

# Constants
INPUT_COLUMNS = [
    'volt1', 'volt2', 'amp1', 'amp2', 'FQtyL', 'FQtyR',
    'E1 FFlow', 'E1 OilT', 'E1 OilP', 'E1 RPM',
    'E1 CHT1', 'E1 CHT2', 'E1 CHT3', 'E1 CHT4',
    'E1 EGT1', 'E1 EGT2', 'E1 EGT3', 'E1 EGT4',
    'OAT', 'IAS', 'VSpd', 'NormAc', 'AltMSL'
]

# class BaseAnomalyDataset(torch.utils.data.Dataset):
#     """Base class for anomaly detection datasets."""
    
#     def __init__(self, data: torch.Tensor, labels: Optional[torch.Tensor] = None):
#         """
#         Initialize the dataset.
        
#         Args:
#             data (torch.Tensor): Input data tensor
#             labels (torch.Tensor, optional): Labels tensor if available
#         """
#         self.data = data
#         self.labels = labels

#     def __len__(self) -> int:
#         """Return the total number of samples."""
#         return len(self.data)

#     def __getitem__(self, idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
#         """
#         Get a sample from the dataset.
        
#         Args:
#             idx (int): Index of the sample
            
#         Returns:
#             Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: 
#                 Either just the data tensor or a tuple of (data, label)
#         """
#         if self.labels is not None:
#             return self.data[idx], self.labels[idx]
#         return self.data[idx]

class DataLoading:
    """Class for loading and preprocessing flight data """
    
    def __init__(self: str):
        """
        Initialize the DataLoading class.
        
        Args:
            data_path (str): Path to the data file
        """

        self.df: Optional[pd.DataFrame] = None

    def load_json(self, filepath: str) -> dict:
        """
        Load data from a JSON file.
        
        Args:
            filepath (str): Path to the JSON file """
        with open(filepath, 'r') as f:
            data = json.load(f)
        return data

    def load_data(self, filepath=None) -> pd.DataFrame:
        """Memory efficient loading"""
        # Read only needed columns
       
        required_columns = INPUT_COLUMNS + ['id', 'split', 'before_after']
        # Use efficient dtypes
        # dtype_dict = {col: 'float32' for col in INPUT_COLUMNS}
        tqdm.pandas(desc="Loading CSV")
        df = pd.read_csv(
            filepath,
            engine='c'
        ).progress_apply(lambda x: x)
        
        df = df.replace(['NONE', 'none'], np.nan)
        # df = df.dropna()
        return df  # Don't store in state

    def process_in_chunks(self, filepath=None, chunksize=10000):
        """For very large datasets"""
        for chunk in pd.read_csv(filepath, chunksize=chunksize):
            processed_chunk = self.preprocess_chunk(chunk)
            yield processed_chunk
    
    def load_data_config(self,data_folder_path, config: dict) -> pd.DataFrame:
        '''
        Load data from a config file 
        
        '''

        data = []
        for event in config:
            for file in event['before_flights']:
                filepath = f"{data_folder_path}/{file}"
                data.append(self.load_data(filepath))

            for file in event['after_flights']:
                data.append(self.load_data(file))
        


    
    def min_max_scaling(self, input_columns: List[str], df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Apply Min-Max scaling to specified columns while preserving other columns.

        Args:
            input_columns (List[str]): List of column names to scale.
            df (Optional[pd.DataFrame]): DataFrame to scale, uses self.df if None.

        Returns:
            pd.DataFrame: DataFrame with scaled values for input columns and original values for other columns.
        """
        df = self.df if df is None else df
        
        qt = preprocessing.MinMaxScaler()
        qt.fit(df.loc[:, input_columns].sample(100000, random_state = 0 ))

        arr = df.loc[:, input_columns].values
        res = qt.transform(arr)

        for i, col in tqdm(enumerate(input_columns)):
            df.loc[:, col] = res[:, i]

        print("Scaled data shape:", df.shape)
        return df
    
    def get_folded_datasets(self, MODEL_LOSS_TYPE: str, df: pd.DataFrame, NFOLD: int) -> List[Union[torch.utils.data.TensorDataset, Tuple[torch.utils.data.TensorDataset, torch.utils.data.TensorDataset]]]:
        """
        Create folded datasets for cross-validation. Can be used for both binary and multi-class classification and post processed for anomaly detection.
        
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
    
    def create_anomaly_normal_loaders(self, folded_datasets: List[torch.utils.data.TensorDataset], batch_size: int = 32) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """
        Creates separate DataLoaders for anomaly and normal data.
        
        Args:
            folded_datasets (List[TensorDataset]): List of folded datasets
            batch_size (int): Batch size for the DataLoaders
            
        Returns:
            Tuple[DataLoader, DataLoader]: Tuple containing (anomalies_loader, normal_loader)
        """
        anomalies = []  # post maintenance (1)
        normal_data = []  # pre maintenance (0)
        
        for fold in folded_datasets:
            for data, label in fold:
                if label == 0:
                    normal_data.append(data.permute(1, 0))
                elif label == 1:
                    anomalies.append(data.permute(1, 0))

        anomalies_dataset = torch.stack(anomalies)
        normal_dataset = torch.stack(normal_data)

        anomalies_loader = torch.utils.data.DataLoader(anomalies_dataset, batch_size=batch_size, shuffle=True)
        normal_loader = torch.utils.data.DataLoader(normal_dataset, batch_size=batch_size, shuffle=True)
        
        return anomalies_loader, normal_loader
    
    