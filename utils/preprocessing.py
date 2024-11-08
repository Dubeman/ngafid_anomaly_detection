import pandas as pd
from statsmodels.tsa.api import VAR
import numpy as np

FEATURES = ['volt1', 'volt2', 'amp1', 'amp2', 'FQtyL', 'FQtyR', 'E1 FFlow',
            'E1 OilT', 'E1 OilP', 'E1 RPM', 'E1 CHT1', 'E1 CHT2', 'E1 CHT3',
            'E1 CHT4', 'E1 EGT1', 'E1 EGT2', 'E1 EGT3', 'E1 EGT4', 'OAT', 'IAS',
            'VSpd', 'NormAc', 'AltMSL']

# Pad groups with rows < max_seq_len using VAR
def pad_group_VAR(group, max_seq_len):
    '''
    params:
    group: pd.DataFrame is the group to pad
    max_seq_len: int is the maximum sequence length to pad to


    '''
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


def preprocess_data(data : pd.DataFrame,MAX_SEQ_LEN, group_by='id' , pad_func=pad_group_VAR):
    # Load the dataset
    labels = data['before_after']

    # Define the maximum sequence length
    max_sequence_length = MAX_SEQ_LEN  # Median of the sequence lengths

    # Group the data by 'plane_id'
    grouped_data = data.groupby(group_by, sort=False)
    print(f"Number of groups: {len(grouped_data)}")

    # Since each id has a unique before_after value, store the label for each id in a np array without disturbing the order
    labels = labels.groupby(data['id'], sort=False).first().values    

    # Pad the sequences for each group
    padded_sequences = []
    plane_ids = []
    
    count = 0

    for plane_id, group in grouped_data:
        # print(f"Processing group {count + 1}/{len(grouped_data)} with plane_id: {plane_id}")

        # Ensure each group has a unique before_after value
        assert len(group['before_after'].unique()) == 1, f"Group {plane_id} has more than one unique before_after value"
        
        # Get the subset of the data
        group = group[FEATURES]

        # Truncate the group if necessary
        if len(group) > max_sequence_length:
            group = group.iloc[:max_sequence_length]

        # Pad the group if necessary
        if len(group) < max_sequence_length:
            group = pad_func(group, max_sequence_length)

        padded_sequences.append(group)
        plane_ids.append(plane_id)
        count += 1

    # Convert the list of padded sequences to a 3D numpy array and transpose to get the desired shape
    padded_sequences_3d = np.stack([seq.values.T for seq in padded_sequences])

    # Print the shapes of the first two sequences to verify
    # print(f"Shape of the first sequence: {padded_sequences_3d[0].shape}")
    # print(f"Shape of the second sequence: {padded_sequences_3d[1].shape}")

    # # Verify the shape and data type of the 3D numpy array
    # print(f"Padded sequences 3D shape: {padded_sequences_3d.shape}")
    # print(f"Padded sequences 3D data type: {padded_sequences_3d.dtype}")

    return padded_sequences_3d, labels

if __name__ == "__main__":
    file_path = '/Users/manasdubey2022/Desktop/NGAFID/Codebase/data/planes/cleaned_flights/34_cleaned.csv'
    data = pd.read_csv(file_path)
    max_sequence_length = 9930
    padded_sequences,_ = preprocess_data(data, max_sequence_length)

    #save the padded sequences to a file
    np.save('/Users/manasdubey2022/Desktop/NGAFID/Codebase/data/planes/cleaned_flights/padded_sequences.npy', padded_sequences)