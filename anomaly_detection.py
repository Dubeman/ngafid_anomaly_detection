import argparse
# import mlflow
# import mlflow.pytorch
from utils.preprocessing import preprocess_data, pad_group_constant
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tsai.all import ROCKET, get_ts_dls, create_rocket_features, L, Learner, accuracy, ShowGraph
from sklearn.linear_model import LogisticRegression, RidgeClassifierCV
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

def lin_zero_init(layer):
    if isinstance(layer, nn.Linear):
        nn.init.constant_(layer.weight.data, 0.)
        if layer.bias is not None: nn.init.constant_(layer.bias.data, 0.)

def create_mlp_head(n_vars, n_classes, seq_len):
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(n_vars * seq_len, 512),
        nn.ReLU(),
        nn.Linear(512, n_classes)
    )



def train_model(data, labels, model, max_seq_len, batch_size, n_kernels, kss, classifier_type='logistic', classifier_args=None, log_mlflow=True):
    if classifier_args is None:
        classifier_args = {}

    splits = (list(range(len(data)-18)), list(range(100, len(data))))
    dls = get_ts_dls(data, y=labels, splits=splits, tfms=None, drop_last=False, shuffle_train=False, batch_tfms=None, bs=batch_size)
    X_train, y_train = create_rocket_features(dls.train, model)
    X_valid, y_valid = create_rocket_features(dls.valid, model)

    if classifier_type == 'fastai':
        # Create FastAI classifier head
        model = create_mlp_head(dls.vars, dls.c, dls.len)
        model.apply(lin_zero_init)
        learn = Learner(dls, model, metrics=accuracy, cbs=ShowGraph())
        learn.fit_one_cycle(50, lr_max=1e-4)
        learn.plot_metrics()
        return learn

    best_loss = np.inf
    Cs = classifier_args.get('Cs', [1.0])
    eps = classifier_args.get('eps', 1e-8)
    for i, C in enumerate(Cs):
        f_mean = X_train.mean(axis=0, keepdims=True)
        f_std = X_train.std(axis=0, keepdims=True) + eps
        X_train_tfm2 = (X_train - f_mean) / f_std
        X_valid_tfm2 = (X_valid - f_mean) / f_std

        if classifier_type == 'logistic':
            classifier = LogisticRegression(penalty='l2', C=C, n_jobs=-1)
        elif classifier_type == 'ridge':
            classifier = RidgeClassifierCV(alphas=np.logspace(-8, 8, 17), normalize=True)
        else:
            raise ValueError("Unsupported classifier type")

        classifier.fit(X_train_tfm2, y_train)
        probas = classifier.predict_proba(X_train_tfm2) if classifier_type == 'logistic' else classifier.decision_function(X_train_tfm2)
        loss = nn.CrossEntropyLoss()(torch.tensor(probas), torch.tensor(y_train)).item()
        train_score = classifier.score(X_train_tfm2, y_train)
        val_score = classifier.score(X_valid_tfm2, y_valid)
        if loss < best_loss:
            best_eps = eps
            best_C = C
            best_loss = loss
            best_train_score = train_score
            best_val_score = val_score
        print(f"{i:2} eps: {eps:.2E}  C: {C:.2E}  loss: {loss:.5f}  train_acc: {train_score:.5f}  valid_acc: {val_score:.5f}")

        if log_mlflow:
            # Log parameters and metrics
            mlflow.log_param("eps", eps)
            mlflow.log_param("C", C)
            mlflow.log_metric("loss", loss)
            mlflow.log_metric("train_acc", train_score)
            mlflow.log_metric("valid_acc", val_score)

    if log_mlflow:
        # Log the best model
        mlflow.sklearn.log_model(classifier, "model")

    return best_eps, best_C, best_loss, best_train_score, best_val_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model on the plane_34 dataset")
    parser.add_argument('--classifier', type=str, choices=['logistic', 'ridge', 'fastai'], default='logistic', help='Classifier type to use')
    parser.add_argument('--Cs', type=float, nargs='+', default=[0.01, 0.1, 1, 10, 100], help='List of C values for logistic regression')
    parser.add_argument('--eps', type=float, default=1e-8, help='Epsilon value for normalization')
    parser.add_argument('--log_mlflow', action='store_true', help='Flag to log information in MLflow')
    args = parser.parse_args()

    if args.log_mlflow:
        mlflow.set_tracking_uri("file:///Users/manasdubey2022/Desktop/NGAFID/Codebase/mlruns")
        mlflow.set_experiment("plane_forecasting")

    plane_34 = pd.read_csv('/Users/manasdubey2022/Desktop/NGAFID/Codebase/data/planes/cleaned_flights/34_cleaned.csv')
    forecast = Forecasting(plane_34)
    max_seq_len = 9930
    data, labels = forecast.preprocess_data(max_seq_len)
    print(f"Data shape: {data.shape}")

    model = ROCKET(c_in=23, seq_len=max_seq_len, n_kernels=10000, kss=[7, 9, 11])
    batch_size = 32
    n_kernels = 10000
    kss = [7, 9, 11]
    classifier_args = {
        'Cs': args.Cs,
        'eps': args.eps
    }

    classifier_type = args.classifier

    if classifier_type == 'fastai':
        learn = train_model(data, labels, model, max_seq_len, batch_size, n_kernels, kss, classifier_type, classifier_args, log_mlflow=args.log_mlflow)
    else:
        best_eps, best_C, best_loss, best_train_score, best_val_score = train_model(data, labels, model, max_seq_len, batch_size, n_kernels, kss, classifier_type, classifier_args, log_mlflow=args.log_mlflow)
        print('\nBest result:')
        print('eps: {:.2E}  C: {:.2E}  train_loss: {:.5f}  train_acc: {:.5f}  valid_acc: {:.5f}'.format(best_eps, best_C, best_loss, best_train_score, best_val_score))