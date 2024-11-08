import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
# from minisom import MiniSom
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import mlflow
from utils.mlflow_logger import log_mlflow_run
from mlflow.models.signature import infer_signature
import joblib 


class AnomalyDetection:
    def __init__(self, data=None):
        self.data = data
        self.scaler = StandardScaler()
        self.ocsvm =  OneClassSVM(kernel='rbf', nu=0.01, gamma=0.01)
        self.isolation_forest = IsolationForest(contamination=0.01)
        self.X_train_features = None
        self.y_train = None
        self.X_val_features = None
        self.y_val = None

    def preprocess_data(self):
        # Fill missing values with mean
        preprocessed_data = self.data.fillna(self.data.mean())
        return preprocessed_data
    
    def read_and_split_rocket_features(self, num_kernels=1000, features_file=None):
        self.num_kernels = num_kernels
         
        X_train_features = pd.read_csv(f"data/rocket_features/X_train_features_rocket_{num_kernels}.csv")
        #print shape of X_train_features
        print(X_train_features.shape)


        #split the data into a train and validation set
        X_train, X_val = train_test_split(X_train_features, test_size=0.2, random_state=42)
        print("Shape of the training set: ", X_train.shape)
        print(X_train.shape)
        print("Shape of the validation set: ", X_val.shape)
        print(X_val.shape)


        # make the features and labels by dropping the last column
        self.X_train_features = X_train.iloc[:, :-1].values
        self.y_train = X_train.iloc[:, -1].values
        self.X_val_features = X_val.iloc[:, :-1].values
        self.y_val = X_val.iloc[:, -1].values

    def dbscan_anomaly_detection(self, eps=0.5, min_samples=5):
        # Preprocess data
        preprocessed_data = self.preprocess_data()
        
        # Standardize the features
        X = self.scaler.fit_transform(preprocessed_data)
        
        # Apply DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X)
        
        # Identify anomalies (points labeled as -1)
        anomalies = preprocessed_data[labels == -1]
        return anomalies, labels

    def som_anomaly_detection(self, x_dim=10, y_dim=10, sigma=1.0, learning_rate=0.5, num_iterations=1000):
        # Preprocess data
        preprocessed_data = self.preprocess_data()
        
        # Standardize the features
        X = self.scaler.fit_transform(preprocessed_data)
        
        # Train SOM
        som = MiniSom(x_dim, y_dim, X.shape[1], sigma=sigma, learning_rate=learning_rate)
        som.random_weights_init(X)
        som.train_random(X, num_iterations)
        
        # Identify anomalies based on distance to BMU (Best Matching Unit)
        distances = np.array([som.winner(x) for x in X])
        threshold = np.percentile(distances, 95)  # Set threshold for anomaly detection
        anomalies = preprocessed_data[distances > threshold]
        return anomalies, distances

    def anomaly_detection(self, model, model_name, experiment_name, log_model=False, registered_model_name=None):
        # Set the MLflow tracking URI to a local directory
        mlflow.set_tracking_uri("file:///Users/manasdubey2022/Desktop/NGAFID/Codebase/mlruns")
    
        mlflow.set_experiment(experiment_name=experiment_name)

        # Start an MLflow run
        with mlflow.start_run():
            # Log the model name
            mlflow.log_param("model_name", model_name)

            # Log model parameters using get_params
            params = model.get_params(deep=True)
            for param_name, param_value in params.items():
                mlflow.log_param(param_name, param_value)

            # Log data split dimensions and number of kernels used
            mlflow.log_param("train_set_size", self.X_train_features.shape[0])
            mlflow.log_param("val_set_size", self.X_val_features.shape[0])
            mlflow.log_param("num_kernels", self.num_kernels)

            # Fit the model and make predictions
            if isinstance(model, DBSCAN):
                y_pred_train = model.fit_predict(self.X_train_features)
                y_pred_val = model.fit_predict(self.X_val_features)
            else:
                model.fit(self.X_train_features)
                y_pred_train = model.predict(self.X_train_features)
                y_pred_val = model.predict(self.X_val_features)

            # Convert cluster labels to binary labels for DBSCAN
            if isinstance(model, DBSCAN):
                y_pred_train_binary = np.where(y_pred_train == -1, 1, 0)
                y_pred_val_binary = np.where(y_pred_val == -1, 1, 0)
            else:
                y_pred_train_binary = np.where(y_pred_train == 1, 0, 1)
                y_pred_val_binary = np.where(y_pred_val == 1, 0, 1)

            # Calculate metrics for the train set
            train_class_report = classification_report(self.y_train, y_pred_train_binary, output_dict=True)

            # Calculate metrics for the validation set
            val_class_report = classification_report(self.y_val, y_pred_val_binary, output_dict=True)

            # Log metrics
            mlflow.log_metric("train_accuracy", train_class_report["accuracy"])
            mlflow.log_metric("val_accuracy", val_class_report["accuracy"])

            with open("train_classification_report.txt", "w") as f:
                f.write(classification_report(self.y_train, y_pred_train_binary))
            with open("val_classification_report.txt", "w") as f:
                f.write(classification_report(self.y_val, y_pred_val_binary))
            mlflow.log_artifact("train_classification_report.txt")
            mlflow.log_artifact("val_classification_report.txt")

            # Optionally, log the model
            if log_model and registered_model_name:
                signature = infer_signature(self.X_train_features, y_pred_train_binary)
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="model",
                    signature=signature,
                    registered_model_name=registered_model_name
                )

        # Print metrics
        print("Train set:")
        print(classification_report(self.y_train, y_pred_train_binary))

        print("Validation set:")
        print(classification_report(self.y_val, y_pred_val_binary))


            



        

# Example usage
if __name__ == "__main__":

# Create an instance of the class containing the anomaly_detection method
    anomaly_detector = AnomalyDetection()

    # Read and split the Rocket features
    anomaly_detector.read_and_split_rocket_features(num_kernels=1000)



    # anomaly_detector.anomaly_detection(ocsvm, "OCSVM")

    model_name = "DBSCAN"
    experiment_name = f"{model_name} Experiments"
    num_kernels = 1000
    registered_model_name = f"{model_name}_model_{num_kernels}"

    # DBSCAN model
    dbscan = DBSCAN(eps=0.5, min_samples=5)

    # Run anomaly detection
    anomaly_detector.anomaly_detection(dbscan, model_name, experiment_name, log_model=True, registered_model_name=registered_model_name)




    
