import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class ChunkedCSVDataset(Dataset):
    def __init__(self, file_path, target_column, chunk_size=10000, transform=None):
        self.file_path = file_path
        self.target_column = target_column
        self.chunk_size = chunk_size
        self.transform = transform
        
        # Read the data in chunks and store them
        self.chunks = list(pd.read_csv(file_path, chunksize=chunk_size))
        self.num_chunks = len(self.chunks)
        
        # Calculate total number of rows
        self.total_rows = sum(len(chunk) for chunk in self.chunks)

    def __len__(self):
        return self.total_rows

    def __getitem__(self, idx):
        # Identify which chunk and row within the chunk to fetch
        chunk_idx = idx // self.chunk_size
        row_idx = idx % self.chunk_size

        # Fetch the appropriate chunk
        chunk = self.chunks[chunk_idx]

        # Safeguard against accessing non-existent rows in the last chunk
        if row_idx >= len(chunk):
            return None  # Return None if the index exceeds the number of rows in the chunk

        # Get the row from the chunk
        row = chunk.iloc[row_idx]

        # Separate features (X) and target (y)
        X = row.drop(self.target_column).values
        y = row[self.target_column]

        # Apply any transforms if provided
        if self.transform:
            X = self.transform(X)
            # Ensure X is a NumPy array
            if isinstance(X, torch.Tensor):
                X = X.numpy()

        X_tensor = torch.from_numpy(X).float().clone().detach()
        y_tensor = torch.tensor(y).long().clone().detach()

        return X_tensor, y_tensor # Return the features and target as PyTorch tensors

def main():
    chunk_size = 10000

    # model_path = 'models/rocket_ridge_800.pkl'
    # save_file = 'results/ridge_classification_report_500.csv'
    batch_size = 512

    def add_channel_dimension(num_channels):
        def transform(X):
            X = torch.tensor(X, dtype=torch.float32)
            return X.unsqueeze(0) if num_channels == 1 else X.view(num_channels, -1)
        return transform

    num_channels = 1

    train_dataset = ChunkedCSVDataset('/shared/rc/gamts/Codebase/data/NGAFID_train_data.csv', 'before_after', chunk_size=chunk_size, transform=add_channel_dimension(num_channels))
    # val_dataset = ChunkedCSVDataset('data/NGAFID_val_data.csv', 'before_after', chunk_size=chunk_size)
    # test_dataset = ChunkedCSVDataset('data/NGAFID_test_data.csv', 'before_after', chunk_size=chunk_size)

    # get the last chunk from the self.chunks list
    print(train_dataset[0][0].shape, train_dataset[0][1].shape)

if __name__ == "__main__":
    main()






