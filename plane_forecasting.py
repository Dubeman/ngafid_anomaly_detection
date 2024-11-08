import mlflow
import mlflow.sklearn
from utils.preprocessing import preprocess_data, pad_group_constant
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tsai.all import ROCKET, get_ts_dls, create_rocket_features, L 
from sklearn.linear_model import LogisticRegression
import torch
import numpy as np
from torch import nn
from sklearn.model_selection import train_test_split

class Forecasting:
    def __init__(self, data, model=None):
        self.data = data
        self.model = model

    def preprocess_data(self, MAX_SEQ_LEN):
        padded_sequences, labels = preprocess_data(self.data, MAX_SEQ_LEN, group_by='id', pad_func=pad_group_constant)
        return padded_sequences, labels

    def scale_data(self, data):
        return self.scaler.fit_transform(data)

def train_model(data, labels, model, max_seq_len, batch_size, n_kernels, kss, Cs, eps):
    splits = (list(range(len(data)-18)), list(range(100, len(data))))
    dls = get_ts_dls(data, y=labels, splits=splits, tfms=None, drop_last=False, shuffle_train=False, batch_tfms=None, bs=batch_size)
    X_train, y_train = create_rocket_features(dls.train, model)
    X_valid, y_valid = create_rocket_features(dls.valid, model)

    best_loss = np.inf
    for i, C in enumerate(Cs):
        f_mean = X_train.mean(axis=0, keepdims=True)
        f_std = X_train.std(axis=0, keepdims=True) + eps
        X_train_tfm2 = (X_train - f_mean) / f_std
        X_valid_tfm2 = (X_valid - f_mean) / f_std
        classifier = LogisticRegression(penalty='l2', C=C, n_jobs=-1)
        classifier.fit(X_train_tfm2, y_train)
        probas = classifier.predict_proba(X_train_tfm2)
        loss = nn.CrossEntropyLoss()(torch.tensor(probas), torch.tensor(y_train)).item()
        train_score = classifier.score(X_train_tfm2, y_train)
        val_score = classifier.score(X_valid_tfm2, y_valid)
        if loss < best_loss:
            best_eps = eps
            best_C = C
            best_loss = loss
            best_train_score = train_score
            best_val_score = val_score
        print('{:2} eps: {:.2E}  C: {:.2E}  loss: {:.5f}  train_acc: {:.5f}  valid_acc: {:.5f}'.format(i, eps, C, loss, train_score, val_score))
    return best_eps, best_C, best_loss, best_train_score, best_val_score

if __name__ == "__main__":
    mlflow.start_run()

    plane_34 = pd.read_csv('/Users/manasdubey2022/Desktop/NGAFID/Codebase/data/planes/cleaned_flights/34_cleaned.csv')
    forecast = Forecasting(plane_34)
    max_seq_len = 9930
    data, labels = forecast.preprocess_data(max_seq_len)
    print(f"Data shape: {data.shape}")
    print(f"Labels shape: {labels.shape}")

    model = ROCKET(c_in=23, seq_len=max_seq_len, n_kernels=10, kss=[3, 5, 7])
    batch_size = 10
    Cs = np.logspace(-5, 5, 11)
    eps = 1e-6

    best_eps, best_C, best_loss, best_train_score, best_val_score = train_model(data, labels, model, max_seq_len, batch_size, 10, [3, 5, 7], Cs, eps)

    mlflow.log_param("max_seq_len", max_seq_len)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("n_kernels", 10)
    mlflow.log_param("kss", [3, 5, 7])
    mlflow.log_param("eps", best_eps)
    mlflow.log_param("C", best_C)
    mlflow.log_metric("best_loss", best_loss)
    mlflow.log_metric("best_train_score", best_train_score)
    mlflow.log_metric("best_val_score", best_val_score)

    mlflow.end_run()

    print('\nBest result:')
    print('eps: {:.2E}  C: {:.2E}  train_loss: {:.5f}  train_acc: {:.5f}  valid_acc: {:.5f}'.format(best_eps, best_C, best_loss, best_train_score, best_val_score))
