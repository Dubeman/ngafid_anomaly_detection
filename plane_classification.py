import argparse
import mlflow
import mlflow.pytorch
from utils.preprocessing import preprocess_data, pad_group_constant
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tsai.all import ROCKET, get_ts_dls, create_rocket_features, L, Learner, accuracy, ShowGraph, LinBnDrop, SigmoidRange, Reshape, Categorize, TSStandardize, get_splits
from sklearn.linear_model import LogisticRegression, RidgeClassifierCV
import xgboost as xgb
import torch
import numpy as np
from torch import nn
from sklearn.model_selection import train_test_split
from torch.nn import CrossEntropyLoss
import matplotlib.pyplot as plt
import json

class Forecasting:
    def __init__(self, data, model=None):
        self.data = data
        self.model = model

    def preprocess_data(self, MAX_SEQ_LEN):
        padded_sequences, labels = preprocess_data(self.data, MAX_SEQ_LEN, group_by='id', pad_func=pad_group_constant)
        return padded_sequences, labels

    def create_dataloaders(self, data, labels, batch_size):
        # Create splits using get_splits
        splits = get_splits(labels, valid_size=0.15, test_size=0.15, shuffle=False, stratify=False, random_state=23, show_plot=False)

        # Print the splits information
        print(f"Train split: {len(splits[0])}, Valid split: {len(splits[1])}, Test split: {len(splits[2])}")

        # Print the min and max of the splits
        print(f"Train split: {min(splits[0])}, {max(splits[0])}, Valid split: {min(splits[1])}, {max(splits[1])}, Test split: {min(splits[2])}, {max(splits[2])}")

        # Create DataLoaders
        tfms = [None, Categorize()]
        batch_tfms = [TSStandardize(by_sample=True)]
        dls = get_ts_dls(data, y=labels, splits=splits[:2], tfms=tfms, drop_last=False, shuffle_train=False, batch_tfms=batch_tfms, bs=batch_size)

        test_dls = get_ts_dls(data, y=labels, splits=splits[2:], tfms=tfms, drop_last=False, shuffle_train=False, batch_tfms=batch_tfms, bs=batch_size)
        test_dl = test_dls.train
        

        return dls , test_dl

def lin_zero_init(layer):
    if isinstance(layer, nn.Linear):
        nn.init.constant_(layer.weight.data, 0.)
        if layer.bias is not None: nn.init.constant_(layer.bias.data, 0.)

def create_mlp_head(nf, c_out, seq_len=None, flatten=True, fc_dropout=0., bn=False, lin_first=False, y_range=None):
    if flatten: nf *= seq_len
    layers = [Reshape()] if flatten else []
    layers += [LinBnDrop(nf, c_out, bn=bn, p=fc_dropout, lin_first=lin_first)]
    if y_range: layers += [SigmoidRange(*y_range)]
    return nn.Sequential(*layers)

def save_metrics(metrics, filename):
    with open(filename, 'w') as f:
        json.dump(metrics, f)

def load_metrics(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def plot_metrics(metrics):
    epochs = range(len(metrics['train_loss']))
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, metrics['train_loss'], label='Train Loss')
    plt.plot(epochs, metrics['valid_loss'], label='Valid Loss')
    plt.plot(epochs, metrics['accuracy'], label='Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Metrics')
    plt.legend()
    plt.title('Training Metrics')
    plt.show()

def train_model(dls, model, classifier_type='logistic', log_mlflow=True):
    X_train, y_train = create_rocket_features(dls.train, model)
    X_valid, y_valid = create_rocket_features(dls.valid, model)

    #print the shapes of the training and validation data
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_valid shape: {X_valid.shape}, y_valid shape: {y_valid.shape}")

    if classifier_type == 'fastai':
        # Create FastAI classifier head
        model = create_mlp_head(dls.vars, dls.c, dls.len)
        model.apply(lin_zero_init)
        learn = Learner(dls, model, loss_func=CrossEntropyLoss(), metrics=accuracy)
        learn.fit(100, lr=1e-3)

        # Extract the training and validation losses
        train_losses = [v[0] for v in learn.recorder.values]
        valid_losses = [v[1] for v in learn.recorder.values]
        
        # Save metrics
        metrics = {
            'train_loss': train_losses,
            'valid_loss': valid_losses,
            'accuracy': [v[2] for v in learn.recorder.values]
        }
        save_metrics(metrics, 'metrics.json')
        
        return learn

    # Normalize the data
    f_mean = X_train.mean(axis=0, keepdims=True)
    f_std = X_train.std(axis=0, keepdims=True) + 1e-8
    X_train_tfm2 = (X_train - f_mean) / f_std
    X_valid_tfm2 = (X_valid - f_mean) / f_std

    if classifier_type == 'logistic':
        classifier = LogisticRegression(penalty='l2', n_jobs=-1)
    elif classifier_type == 'ridge':
        classifier = RidgeClassifierCV(alphas=np.logspace(-8, 8, 17))
    elif classifier_type == 'xgboost':
        classifier = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    else:
        raise ValueError("Unsupported classifier type")

    classifier.fit(X_train_tfm2, y_train)
    train_score = classifier.score(X_train_tfm2, y_train)
    val_score = classifier.score(X_valid_tfm2, y_valid)
    print(f"train_acc: {train_score:.5f}  valid_acc: {val_score:.5f}")

    # Get predictions
    y_train_pred = classifier.predict(X_train_tfm2)
    y_valid_pred = classifier.predict(X_valid_tfm2)

    # Print actual and predicted labels
    # print("Actual vs Predicted labels (Train):")
    # for actual, predicted in zip(y_train[:10], y_train_pred[:10]):
    #     print(f"Actual: {actual}, Predicted: {predicted}")

    # print("Actual vs Predicted labels (Validation):")
    # for actual, predicted in zip(y_valid[:10], y_valid_pred[:10]):
    #     print(f"Actual: {actual}, Predicted: {predicted}")

    if log_mlflow:
        # Log the best model
        mlflow.sklearn.log_model(classifier, "model")

    return train_score, val_score, classifier

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model on the plane_34 dataset")

    parser.add_argument('--num_kernels', type=int, default=1000, help='Number of kernels to use in the ROCKET model')
    parser.add_argument('--classifier', type=str, choices=['logistic', 'ridge', 'fastai', 'xgboost'], default='logistic', help='Classifier type to use')
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

    batch_size = 32
    n_kernels = args.num_kernels
    kss = [7, 9, 11]

    model = ROCKET(c_in=23, seq_len=max_seq_len, n_kernels=n_kernels, kss=kss)

    classifier_type = args.classifier

    dls, test_dl = forecast.create_dataloaders(data, labels, batch_size)




    if classifier_type == 'fastai':
        learn  = train_model(dls, model, classifier_type, log_mlflow=args.log_mlflow)
    else:
        train_score, val_score, classifier = train_model(dls, model, classifier_type, log_mlflow=args.log_mlflow)
        print('\nBest result:')
        print(f'train_acc: {train_score:.5f}  valid_acc: {val_score:.5f}')

    # Plot metrics after training
    if classifier_type == 'fastai':
        metrics = load_metrics('metrics.json')
        plot_metrics(metrics)

    # Evaluate on test set
    #print the shapes of the test_dl
    xb,yb = next(iter(test_dl))
    print(f"Xb shape: {xb.shape}, yb shape: {yb.shape}")

    if classifier_type == 'fastai':
        test_score = learn.validate(dl=test_dl)
        print(f'test_acc: {test_score[1]:.5f}')
    else:

        X_test, y_test = create_rocket_features(test_dl, model)
        test_score = classifier.score(X_test, y_test)
        print(f'test_acc: {test_score:.5f}')