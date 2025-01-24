import pandas as pd
import numpy as np
from sklearn import preprocessing 
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from utils.preprocessing import get_dataset, prepare_for_training
import torch

INPUT_COLUMNS = ['volt1',
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



class DataLoading:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None # original dataframe

    def load_data(self):
        df_test = pd.read_csv(self.data_path, nrows=100)

        float_cols = [c for c in df_test if df_test[c].dtype == "float64"]
        # Change float32_cols to use float32 instead of float16
        float32_cols = {c: np.float32 for c in float_cols}

        df = pd.read_csv(self.data_path, engine='c', dtype=float32_cols)
        df['id'] = df.id.astype('int32')
        self.df = df.dropna() # you can handle nans differently, but ymmv
        print(df.head(5))

        return df
    
    def min_max_scaling(self, input_columns, df=None):
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

            raise  # Re-raise the original exception after printing the information

        arr = df.loc[:, input_columns].values
        res = qt.transform(arr)

        for i, col in tqdm(enumerate(input_columns)):
            df.loc[:, col] = res[:, i]

        return df
    
    def get_folded_datasets(self,MODEL_LOSS_TYPE,df,NFOLD):
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
    


    

    

    


    



def main():
        data_path = "data/NGAFID_MC_C37.csv"
        data_loading = DataLoading(data_path)
        data = data_loading.load_data()
        data = data_loading.min_max_scaling(INPUT_COLUMNS)
        print(data.head(5))

        folded_datasets = data_loading.get_folded_datasets('bce',data,5)
        print(type(folded_datasets[0]))
        print(folded_datasets[0])

if __name__ == "__main__":
    main()
    

