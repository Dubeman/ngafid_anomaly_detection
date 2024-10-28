import pandas as pd
from sklearn.preprocessing import StandardScaler

'''
Here we are going to split the data into training, validation, and testing sets for time series data.
This is done by splitting the data into 70% training, 15% validation, and 15% testing sets.
Scaling is performed on the training set and then applied to the validation and testing sets.
The scaled datasets are saved into separate files.
'''



# Function to concatenate DataFrames in chunks
def concat_and_save_in_chunks(df1, df2, file_path, chunk_size=10000):
    with open(file_path, 'w') as f:
        for i in range(0, len(df1), chunk_size):
            chunk = pd.concat([df1.iloc[i:i+chunk_size], df2.iloc[i:i+chunk_size].reset_index(drop=True)], axis=1)
            chunk.to_csv(f, header=(i==0), index=False)


# Load the dataset
data_imputed = pd.read_csv('/shared/rc/gamts/Codebase/data/NGAFID_imputed_data.csv')

# Separate the 'before_after' column (target) from the other features
before_after = data_imputed['before_after']
features = data_imputed.drop(columns=['before_after'])

# Add the 'before_after' column back to the features DataFrame for grouping
features['before_after'] = before_after

# Group the data by 'plane_id'
grouped = features.groupby('plane_id')

# Convert the grouped object to a list of DataFrames
grouped_list = [group for _, group in grouped]

# Determine split points for training (70%), validation (15%), and testing (15%)
total_groups = len(grouped_list)
train_size = int(0.7 * total_groups)
val_size = int(0.15 * total_groups)

# Split the groups into training, validation, and testing sets
train_groups = grouped_list[:train_size]
val_groups = grouped_list[train_size:train_size + val_size]
test_groups = grouped_list[train_size + val_size:]

# #make sure that the 'plane_id' columns are unique in each set
# train_plane_ids = set()
# val_plane_ids = set()
# test_plane_ids = set()

# for group in train_groups:
#     train_plane_ids.update(group['plane_id'].unique())

# for group in val_groups:
#     val_plane_ids.update(group['plane_id'].unique())

# for group in test_groups:
#     test_plane_ids.update(group['plane_id'].unique())

# # Check if there are any common 'plane_id' values between the sets, should return False
# common_ids = train_plane_ids.intersection(val_plane_ids).union(train_plane_ids.intersection(test_plane_ids)).union(val_plane_ids.intersection(test_plane_ids))

# if common_ids:
#     print("There are common 'plane_id' values between the sets.")
# else:
#     print("There are no common 'plane_id' values between the sets.")

# Concatenate the groups back into DataFrames
train_data = pd.concat(train_groups)
val_data = pd.concat(val_groups)
test_data = pd.concat(test_groups)

# Separate the features and target again
X_train = train_data.drop(columns=['before_after'])
y_train = train_data['before_after']

X_val = val_data.drop(columns=['before_after'])
y_val = val_data['before_after']

X_test = test_data.drop(columns=['before_after'])
y_test = test_data['before_after']


# also drop the 'plane_id' column, 'id' column, 'split' and 'date_diff'
X_train = X_train.drop(columns=['plane_id', 'id', 'split', 'date_diff'])
X_val = X_val.drop(columns=['plane_id', 'id', 'split', 'date_diff'])
X_test = X_test.drop(columns=['plane_id', 'id', 'split', 'date_diff'])

# Perform feature-wise scaling on the training features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Transform the validation and test features using the same scaler
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)


# Convert the scaled features back to DataFrames
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_val_scaled_df = pd.DataFrame(X_val_scaled, columns=X_val.columns)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# Save the scaled training, validation, and testing sets into separate files in chunks
concat_and_save_in_chunks(X_train_scaled_df, y_train, '/shared/rc/gamts/Codebase/data/NGAFID_train_data.csv')
concat_and_save_in_chunks(X_val_scaled_df, y_val, '/shared/rc/gamts/Codebase/data/NGAFID_val_data.csv')
concat_and_save_in_chunks(X_test_scaled_df, y_test, '/shared/rc/gamts/Codebase/data/NGAFID_test_data.csv')


print("Training, validation, and testing sets have been saved.")