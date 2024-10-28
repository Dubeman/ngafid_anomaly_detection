'''

Pipeline for training and evaluating a ROCKET model for time series forecasting through batch processing.
'''


import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import torch.optim as optim
import mlflow
import mlflow.pytorch
import joblib
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from utils.ChunkedCSVLoader import ChunkedCSVDataset
from tqdm import tqdm
from ROCKET import *
from rocket.code import *
from sklearn.svm import OneClassSVM
# import time
# from sktime.classification.kernel_based import RocketClassifier
import os

class RocketForecasting:

    def __init__(self, train_loader, val_loader, test_loader, num_kernels=10_000, model_path=None, classifier=None):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.scaler = StandardScaler()
        self.rocket_feature_extractor = ROCKET(c_in = 1, seq_len = 27 , n_kernels=num_kernels, kss= [7, 9, 11] )
        self.num_kernels = num_kernels
        self.kernels = None
        
        if model_path:
            self.classifier = joblib.load(model_path)
            print("Loaded model from file.")
        else:
            self.classifier = classifier

    def ROCKET_evaluate(self, test=False, return_metrics=False, model_path=None):
        if test:
            self.rocket_classifier = joblib.load(model_path)
        
        val_loader = self.val_loader if not test else self.test_loader
        all_predictions = []
        all_true_labels = []
        
        total_batches = len(val_loader)
        for batch_idx, (X_batch, y_batch) in enumerate(tqdm(val_loader, disable=not test, desc="Processing Batches")):
            X_batch = X_batch.numpy()
            y_batch = y_batch.numpy()
            y_pred = self.rocket_classifier.predict(X_batch)
            all_predictions.extend(y_pred)
            all_true_labels.extend(y_batch)

            if test:
                batch_accuracy = accuracy_score(y_batch, y_pred)
                print(f"--- Batch {batch_idx + 1} / {total_batches} - Batch Accuracy: {batch_accuracy:.4f} ---")
        
        if return_metrics:
            report = classification_report(all_true_labels, all_predictions, output_dict=True)
            test_accuracy = report['accuracy']
        else:
            report = None
            test_accuracy = accuracy_score(all_true_labels, all_predictions)
        return test_accuracy, report
    
    def Rocket_deep(self, model_path, epochs):
        for epoch in range(epochs):
            print(f"Epoch: {epoch + 1}/{epochs}----------------")

            # Reload the latest model if it exists
            try:
                self.classifier = joblib.load(model_path)
                print(f"Model loaded from {model_path}")
            except FileNotFoundError:
                print(f"No existing model found at {model_path}. Starting fresh.")

            total_batches = len(self.train_loader)
            for batch_idx, (X_batch, y_batch) in enumerate(self.train_loader):
                X_batch = X_batch.unsqueeze(1) # because the rocket model expects 3D input
                y_batch = y_batch

                # Extract ROCKET features
                X_features = self.rocket_feature_extractor(X_batch)
                print(f"X_batch_shape{X_batch.shape}, y_batch_shape{y_batch.shape}")
                y_pred = self.classifier.predict(X_batch).astype(int)

                train_accuracy = accuracy_score(y_batch, y_pred)
                print(f"Epoch {epoch + 1}/{epochs}, Batch {batch_idx + 1}/ {total_batches} - Train Accuracy: {train_accuracy:.4f}")

                # Save the model every 5 batches
                if (batch_idx + 1) % 5 == 0:
                    joblib.dump(self.rocket_classifier, model_path)
                    print(f"Model saved at {model_path}")

            # Validation accuracy
            print("Validation started....")
            val_accuracy, _ = self.ROCKET_evaluate()
            print(f"Epoch {epoch + 1}/{epochs} - Validation Accuracy: {val_accuracy:.4f}")
            print("Validation done....")

        return self.rocket_classifier
    
    def extract_and_save_features(self):


        train_feature_and_labels_path = f'data/rocket_features/X_train_features_rocket_{self.num_kernels}.csv'
        val_features_and_labels_path = f'data/rocket_features/X_val_features_rocket_{self.num_kernels}.csv'
        test_features_and_labels_path = f'data/rocket_features/X_test_features_rocket_{self.num_kernels}.csv'

                    # Extract features from the train_loader and val_loader and save them to disk
        print("Extracting and saving train features....")
        create_rocket_features(train_loader, self.rocket_feature_extractor, train_feature_and_labels_path)
        print("Extracting and saving val features....")
        create_rocket_features(val_loader, self.rocket_feature_extractor, val_features_and_labels_path)
        print("Extracting and saving test features....")
        create_rocket_features(test_loader, self.rocket_feature_extractor,  test_features_and_labels_path)
        print("Features extracted and saved successfully.")



    def ROCKET_ridge_classification(self, model_path, epochs):
        for epoch in range(epochs):
            print(f"Epoch: {epoch + 1}/{epochs}----------------")

            # Reload the latest model if it exists
            try:
                self.rocket_classifier = joblib.load(model_path)
                print(f"Model loaded from {model_path}")
            except FileNotFoundError:
                print(f"No existing model found at {model_path}. Starting fresh.")

            total_batches = len(self.train_loader)
            for batch_idx, (X_batch, y_batch) in enumerate(self.train_loader):
                X_batch = X_batch.numpy()
                y_batch = y_batch.numpy().astype(int)
                self.rocket_classifier.fit(X_batch, y_batch)
                print(f"X_batch_shape{X_batch.shape}, y_batch_shape{y_batch.shape}")
                y_pred = self.rocket_classifier.predict(X_batch).astype(int)

                train_accuracy = accuracy_score(y_batch, y_pred)
                print(f"Epoch {epoch + 1}/{epochs}, Batch {batch_idx + 1}/ {total_batches} - Train Accuracy: {train_accuracy:.4f}")

                # Save the model every 5 batches
                if (batch_idx + 1) % 5 == 0:
                    joblib.dump(self.rocket_classifier, model_path)
                    print(f"Model saved at {model_path}")

            # Validation accuracy
            print("Validation started....")
            val_accuracy, _ = self.ROCKET_evaluate()
            print(f"Epoch {epoch + 1}/{epochs} - Validation Accuracy: {val_accuracy:.4f}")
            print("Validation done....")

        return self.rocket_classifier

    def save_model(self, model_path, feature_extractor_path):
        # Save the feature extractor separately
        torch.save(self.rocket_feature_extractor.state_dict(), feature_extractor_path)
        
        # Save the rest of the model (excluding the feature extractor) using joblib
        joblib.dump(self, model_path)

    @staticmethod
    def load_model(model_path, feature_extractor_path):
        # Load the RocketForecasting object using joblib
        model = joblib.load(model_path)
        
        # Load the feature extractor's state_dict
        model.rocket_feature_extractor.load_state_dict(torch.load(feature_extractor_path))
        
        return model


def add_channel_dimension(num_channels):
    def transform(X):
        X = torch.tensor(X, dtype=torch.float32)
        return X.unsqueeze(0) if num_channels == 1 else X.view(num_channels, -1)
    return transform

def evaluate_anomalies(y_true, y_pred):
    report = classification_report(y_true, y_pred, output_dict=True)
    accuracy = accuracy_score(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)
    return accuracy, report, conf_matrix

def load_features(file_path):
    return pd.read_csv(file_path)


if __name__ == "__main__":
    '''
    1. Load data: Load training, validation, and test datasets.
    2. Initialize RocketForecasting: Create an instance of the RocketForecasting class.
    3. Extract and save features: Use the ROCKET feature extractor to transform the data and save the features.
    4. Train One-Class SVM: Fit the One-Class SVM model using the extracted features.
    5. Evaluate model: Evaluate the trained model on validation data and make predictions on new data.
    '''


    chunk_size = 10000
    num_kernels = 1000
    model_path = f'models/entire_models/rocket_ocsvm_{num_kernels}.pkl'
    feature_extractor_path = f'models/feature_extractors/rocket_{num_kernels}.pkl'   

    batch_size = 512


    train_dataset = ChunkedCSVDataset('data/NGAFID_train_data.csv', 'before_after', chunk_size=chunk_size, transform=add_channel_dimension(1))
    val_dataset = ChunkedCSVDataset('data/NGAFID_val_data.csv', 'before_after', chunk_size=chunk_size, transform=add_channel_dimension(1))
    test_dataset = ChunkedCSVDataset('data/NGAFID_test_data.csv', 'before_after', chunk_size=chunk_size, transform=add_channel_dimension(1))

    train_loader = DataLoader(train_dataset, batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

    rocket_forecasting = RocketForecasting(train_loader, val_loader, test_loader, num_kernels=num_kernels, classifier=OneClassSVM(kernel='rbf', gamma='auto'))


    # save the ROCKET model (feature extractor) to mlflow
    mlflow.pytorch.log_model(rocket_forecasting.rocket_feature_extractor, "rocket_feature_extractor")


    # Extract and save ROCKET features
    rocket_forecasting.extract_and_save_features()
    rocket_forecasting.extract_and_save_features()
    rocket_forecasting.extract_and_save_features()

    ####################### Train the One-Class SVM #######################


    # # Load features
    # X_train_features_and_labels = load_features('data/X_train_features_rocket_800.csv')
    # X_val_features_and_labels = load_features('data/X_val_features_rocket_800.csv')
        
    #     # Remove the labels from the features
    # X_train_features = X_train_features_and_labels.iloc[:, :-1].values

    # #make a ChunkedCSVDataset for the train and val data
    # train_features_dataset = ChunkedCSVDataset('data/NGAFID_train_data.csv', 'before_after', chunk_size=chunk_size)
    # val_features_dataset = ChunkedCSVDataset('data/NGAFID_val_data.csv', 'before_after', chunk_size=chunk_size)


    # # Load the features
    # train_features_loader = DataLoader(train_dataset, batch_size, shuffle=False)
    # val_features_loader = DataLoader(val_dataset, batch_size, shuffle=False)


    # # Fit the One-Class SVM
    # print("Training One-Class SVM....")
    # start_time = time.time()
    # rocket_forecasting.classifier.fit(X_train_features)
    # print(f"Training time: {time.time() - start_time:.2f} seconds")
    # # Save the entire RocketForecasting model
    # rocket_forecasting.save_model(model_path, feature_extractor_path)

    # ####################### Evaluate the One-Class SVM #######################

    # #load the entire model
    # rocket_forecasting = RocketForecasting.load_model(model_path, feature_extractor_path)


    # rocket_forecasting.train_loader = train_features_loader
    # rocket_forecasting.val_loader = val_features_loader
    
    # # Evaluate on validation data
    # val_accuracy, val_report = rocket_forecasting.ROCKET_evaluate(test=False, return_metrics=True)
    # print("Validation Accuracy:", val_accuracy)
    # print("Validation Report:", val_report)
