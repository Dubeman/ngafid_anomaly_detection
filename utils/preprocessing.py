import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from statsmodels.tsa.api import VAR

FEATURES = ['volt1', 'volt2', 'amp1', 'amp2', 'FQtyL', 'FQtyR', 'E1 FFlow',
            'E1 OilT', 'E1 OilP', 'E1 RPM', 'E1 CHT1', 'E1 CHT2', 'E1 CHT3',
            'E1 CHT4', 'E1 EGT1', 'E1 EGT2', 'E1 EGT3', 'E1 EGT4', 'OAT', 'IAS',
            'VSpd', 'NormAc', 'AltMSL']

def pad_group_VAR(group, max_seq_len):
    model = VAR(group)
    model_fit = model.fit(maxlags=1)
    forecast = model_fit.forecast(group.values[-model_fit.k_ar:], steps=max_seq_len - len(group)) # Forecast the next max_seq_len - len(group) steps
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
        extra_tensor = F.interpolate(group_tensor, size=current_seq_len + extra_seq_len, mode=mode, align_corners=False)
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
            group = pad_func(group, max_sequence_length, **pad_kwargs)

        padded_sequences.append(group[FEATURES].values)
        plane_ids.append(plane_id)
        count += 1

    # Convert the list of padded sequences to a 3D numpy array and transpose to get the desired shape
    padded_sequences_3d = np.stack([seq.T for seq in padded_sequences])

    return padded_sequences_3d, labels

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
    padded_sequences_interp, labels_interp = preprocess_data(data, max_seq_len, pad_func=pad_group_interpolate, mode='linear')
    print(f"Padded sequences shape (interpolate): {padded_sequences_interp.shape}")