import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from statsmodels.tsa.api import VAR
import matplotlib.pyplot as plt
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from tqdm import tqdm

FEATURES = ['volt1', 'volt2', 'amp1', 'amp2', 'FQtyL', 'FQtyR', 'E1 FFlow',
            'E1 OilT', 'E1 OilP', 'E1 RPM', 'E1 CHT1', 'E1 CHT2', 'E1 CHT3',
            'E1 CHT4', 'E1 EGT1', 'E1 EGT2', 'E1 EGT3', 'E1 EGT4', 'OAT', 'IAS',
            'VSpd', 'NormAc', 'AltMSL']



SHAPE = (4096, 23)  # Example shape, adjust as needed


def get_dataset(df):
    """
    Processes a DataFrame to create a PyTorch dataset.
    Args:
        df (pandas.DataFrame): The input DataFrame containing sensor data. 
                               It must have columns 'id' and 'before_after'.
    Returns:
        torch.utils.data.TensorDataset: A PyTorch dataset containing the processed sensor data and labels.
    The function performs the following steps:
    1. Extracts unique IDs from the DataFrame.
    2. For each unique ID, extracts the last SHAPE[0] rows of sensor data (first 23 columns).
    3. Pads the sensor data if it has fewer than SHAPE[0] rows.
    4. Converts the sensor data to a PyTorch tensor.
    5. Extracts the 'before_after' label for each ID.
    6. Stacks all sensor data tensors and labels into a single PyTorch dataset.
    """
    ids = df.id.unique()

    sensor_datas = []
    afters = []

    for id in tqdm(ids):
        sensor_data = df[df.id == id].iloc[-SHAPE[0]:, :23].values
        
        sensor_data = np.pad(sensor_data, ((0, SHAPE[0] - len(sensor_data)), (0, 0)))

        # Convert to PyTorch tensor
        sensor_data = torch.tensor(sensor_data, dtype=torch.float32)

        after = df[df.id == id]['before_after'].iloc[0]

        sensor_datas.append(sensor_data)
        afters.append(after)

    # Stack the tensors


    sensor_datas = torch.stack(sensor_datas)


    afters = torch.tensor(afters, dtype=torch.float32)

    # Create a PyTorch dataset
    dataset = torch.utils.data.TensorDataset(sensor_datas, afters)

    return dataset

def fix_type(x, y):
    return x.float(), y.float()

def prepare_for_training(ds, shuffle=False, repeat=False, predict=True, aug=False, batch_size=32, model_type='short_lstm', model_loss_type='bce'):
    """
    Prepares the dataset for training by applying various transformations such as shuffling, repeating, and batching.

    Args:
        ds (Dataset): The dataset to be prepared.
        shuffle (bool, optional): If True, shuffles the dataset. Defaults to False.
        repeat (bool, optional): If True, repeats the dataset. Defaults to False.
        predict (bool, optional): If True, prepares the dataset for prediction. Defaults to True.
        aug (bool, optional): If True, applies data augmentation. Defaults to False.
        batch_size (int, optional): The size of the batches. Defaults to 32.
        model_type (str, optional): The type of model being used. Defaults to 'short_lstm'.
        model_loss_type (str, optional): The type of loss function being used. Defaults to 'bce'.

    Returns:
        DataLoader: The prepared dataset wrapped in a DataLoader.
    """
    if shuffle:
        ds = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)
    else:
        ds = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=True)

    if repeat:
        ds = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=True)

    if not predict:
        ds = [(x, x) for x, y in ds]
    else:
        if model_loss_type == 'bce':
            ds = [(x, y.view(-1, 1)) for x, y in ds]

    return ds


def get_train_and_val_for_fold(folded_datasets, fold, MODEL_LOSS_TYPE, NFOLD, AUGMENT, PREDICT):
    """
    Splits the provided datasets into training and validation sets for a given fold and prepares them for training.

    Args:
        folded_datasets (list): A list of datasets, where each dataset corresponds to a fold.
        fold (int): The index of the fold to be used as the validation set.
        MODEL_LOSS_TYPE (str): The type of model loss ('bce' for binary cross-entropy, 'mse' for mean squared error).
        NFOLD (int): The total number of folds.
        AUGMENT (bool): Whether to apply data augmentation to the training set.
        PREDICT (bool): Whether the datasets are being prepared for prediction.

    Returns:
        tuple: A tuple containing:
            - train_ds: The training dataset prepared for training.
            - val_ds: The validation dataset prepared for training.
            - mse_val_ds: The validation dataset prepared for training if MODEL_LOSS_TYPE is 'mse', otherwise None.
    """

    predict = True


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
        train_ds = ds if train_ds is None else train_ds.concatenate(ds)

    mse_val_ds = None if not MODEL_LOSS_TYPE == 'mse' else prepare_for_training(val_ds, shuffle=False,  predict=True)
    train_ds = prepare_for_training(train_ds, shuffle=True, repeat = True, predict=PREDICT, aug = AUGMENT)
    val_ds = prepare_for_training(val_ds, shuffle=False,  predict=PREDICT)

    return train_ds, val_ds, mse_val_ds


def pad_group_VAR(group, max_seq_len):
    model = VAR(group)
    model_fit = model.fit(maxlags=1)
    forecast = model_fit.forecast(group.values[-model_fit.k_ar:], steps = max_seq_len - len(group)) # Forecast the next max_seq_len - len(group) steps
    forecast_df = pd.DataFrame(forecast, columns=group.columns, index=range(len(group), max_seq_len))
    return pd.concat([group, forecast_df])

def pad_group_constant(group, max_seq_len):
    '''
    params:
    group: pd.DataFrame is the group to pad
    max_seq_len: int is the maximum sequence length to pad to
    this function pads the group with the most repeated value in the group
    '''
    # get the most repeated value in the group
    most_repeated = group.mode().iloc[0]

    # pad the group with the most repeated value uptil max_seq_len
    padding = pd.DataFrame([most_repeated] * (max_seq_len - len(group)), columns=group.columns)
    return pd.concat([group, padding])

def pad_group_interpolate(group, max_seq_len, mode='linear'):
    '''
    params:
    group: pd.DataFrame is the group to pad
    max_seq_len: int is the maximum sequence length to pad to
    mode: str is the interpolation mode to use ('linear', 'nearest', etc.)
    '''

    # Convert to tensor and add batch and channel dimensions
    group_tensor = torch.tensor(group.values).unsqueeze(0).transpose(1, 2)  # Shape: [1, n_features, seq_len]

    # Define the desired sequence length and interpolation mode
    current_seq_len = group_tensor.shape[2]
    extra_seq_len = max_seq_len - current_seq_len

    # Resize to the desired sequence length using interpolation
    if extra_seq_len > 0:
        extra_tensor = F.interpolate(group_tensor, size=current_seq_len + extra_seq_len, mode=mode, align_corners=False)     #
        extra_tensor = extra_tensor[:, :, current_seq_len:]  # Get only the extra observations
        
    else:
        extra_tensor = group_tensor[:, :, :0]  # No extra observations needed

    # Concatenate the original tensor with the extra observations
    resized_tensor = torch.cat((group_tensor, extra_tensor), dim=2)

    # Convert the resized tensor back to a DataFrame
    resized_df = pd.DataFrame(resized_tensor.squeeze().transpose(0, 1).numpy(), columns=group.columns)
    return resized_df




def preprocess_data(data: pd.DataFrame, MAX_SEQ_LEN, group_by='id', pad_func=pad_group_VAR, **pad_kwargs):
    # Load the dataset
    labels = data['before_after']

    # Define the maximum sequence length
    max_sequence_length = MAX_SEQ_LEN  # Median of the sequence lengths

    #data shape
    print(f"Data shape: {data.shape}")

    # Group the data by 'plane_id'
    grouped_data = data.groupby(group_by, sort=False)
    # print(f"Number of groups: {len(grouped_data)}")

    # Since each id has a unique before_after value, store the label for each id in a np array without disturbing the order
    labels = labels.groupby(data['id'], sort=False).first().values    

    # Pad the sequences for each group
    padded_sequences = []
    plane_ids = []
    
    count = 0

    for plane_id, group in grouped_data:
        # Ensure each group has a unique before_after value
        assert len(group['before_after'].unique()) == 1, f"Group {plane_id} has more than one unique before_after value"


        #input group shape is [seq_len, n_features]
        # print(f"Group shape: {group.shape}")

        # Truncate the group if necessary
        if len(group) > max_sequence_length:
            group = group.iloc[:max_sequence_length]

        # Pad the group if necessary
        if len(group) < max_sequence_length:
            print(group["id"])
            group = pad_func(group, max_sequence_length, **pad_kwargs)

        padded_sequences.append(group[FEATURES].values)
        plane_ids.append(plane_id)
        count += 1

    # Convert the list of padded sequences to a 3D numpy array and transpose to get the desired shape
    padded_sequences_3d = np.stack([seq.T for seq in padded_sequences])

    return padded_sequences_3d, labels




def pad_group_dtw(group, max_seq_len):
    '''
    params:
    group: pd.DataFrame is the group to pad
    max_seq_len: int is the maximum sequence length to pad to
    this function interpolates the current group data from its length to the desired max_seq_len using FastDTW
    '''
    group_values = group.values
    current_length = len(group_values)
    target_length = max_seq_len

    # Create a target sequence with the desired length
    target_sequence = np.linspace(0, current_length - 1, target_length)

    # Ensure the target sequence has the same number of features as the group values
    target_sequence = np.tile(target_sequence[:, None], (1, group_values.shape[1]))

    # Perform DTW alignment
    distance, path = fastdtw(group_values, target_sequence, dist=euclidean)

    # Interpolate the group values based on the DTW path
    interpolated_values = np.zeros((target_length, group_values.shape[1]))
    for i, (index_group, index_target) in enumerate(path):
        interpolated_values[index_target] = group_values[index_group]

    # Fill any missing values by linear interpolation
    for i in range(group_values.shape[1]):
        valid_idx = np.where(interpolated_values[:, i] != 0)[0]
        interpolated_values[:, i] = np.interp(np.arange(target_length), valid_idx, interpolated_values[valid_idx, i])

    interpolated_df = pd.DataFrame(interpolated_values, columns=group.columns)
    return interpolated_df

# def interpolate_full_data(data, max_seq_len, mode='linear'):
#     '''
#     Returns the full data with interpolated sequencesq
#     '''
    
    
def plot_padded_values(original_group, padded_group, title):
    num_features = len(FEATURES)
    num_cols = 4
    num_rows = (num_features + num_cols - 1) // num_cols  # Calculate the number of rows needed

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, num_rows * 5))
    axes = axes.flatten()

    for i, feature in enumerate(FEATURES):
        ax = axes[i]
        ax.plot(original_group.index, original_group[feature], label=f'Original {feature}')
        # ax.plot(padded_group.index, padded_group[feature], linestyle='--', label=f'Padded {feature}')
        ax.scatter(padded_group.index[len(original_group):], padded_group[feature][len(original_group):], color='orange', linestyle='--', label='Interpolated Points')
        ax.set_title(f'{title} - {feature}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.legend()

    # Remove any empty subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(f'/Users/manasdubey2022/Desktop/NGAFID/plots/{title}.png')
    plt.show()





# Example usage
if __name__ == "__main__":
    data = pd.read_csv('/Users/manasdubey2022/Desktop/NGAFID/Codebase/data/planes/cleaned_flights/34_cleaned.csv')
    max_seq_len = 9930

    # Using VAR padding
    # padded_sequences_VAR, labels_VAR = preprocess_data(data, max_seq_len, pad_func=pad_group_VAR)
    # print(f"Padded sequences shape (VAR): {padded_sequences_VAR.shape}")

    # Using constant padding
    # padded_sequences_const, labels_const = preprocess_data(data, max_seq_len, pad_func=pad_group_constant)
    # print(f"Padded sequences shape (constant): {padded_sequences_const.shape}")

    # Using interpolation padding
    # padded_sequences_interp, labels_interp = preprocess_data(data, max_seq_len, pad_func=pad_group_interpolate, mode='linear')
    # print(f"Padded sequences shape (interpolate): {padded_sequences_interp.shape}")

    # Print unique values in the 'id' column
    ids = data['id'].unique()


        # Select a group for demonstration 
    group = data.groupby('id').get_group(147) # 147 has less than max_seq_len observations

        # Reset the index of the group
    group = group.reset_index(drop=True)








        # Using linear interpolation padding
    padded_group_interp_linear = pad_group_interpolate(group, max_seq_len, mode='linear')
    padded_group_interp_dtw = pad_group_dtw(group, max_seq_len)


    assert len(padded_group_interp_linear) == max_seq_len, f"Length of the padded group is not equal to {max_seq_len}"

    # Plot the original and padded values
    # plot_padded_values(group, padded_group_interp_linear, 'Linear Interpolation Padding')
    plot_padded_values(group, padded_group_interp_dtw, 'FastDTW Interpolation Padding')

    #plot the 'AltMSL' feature for all group_ids in the non-interpolated original data and save the plots to a folder
    all_groups = data.groupby('id')
    num_ids = 10
    num_cols = 4
    num_rows = (num_ids + num_cols - 1) // num_cols  # Calculate the number of rows needed

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, num_rows * 5))
    axes = axes.flatten()

    for i, id in enumerate(ids[:num_ids]):
        group = all_groups.get_group(id)
        group = group.reset_index(drop=True)
        ax = axes[i]
        ax.plot(group.index, group['AltMSL'])
        ax.set_title(f'AltMSL for Group {id} flight_id_34')
        ax.set_xlabel('Time')
        ax.set_ylabel('Altitude (ft)')

    # Remove any empty subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig('/Users/manasdubey2022/Desktop/NGAFID/plots/AltMSL_all_ids.png')
    plt.show()

    #plot the 'AltMSL' feature for all group_ids in the interpolated data and save the plots to a folder