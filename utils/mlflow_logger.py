import mlflow
import numpy as np
from sklearn.metrics import classification_report
from mlflow.models.signature import infer_signature


def log_mlflow_run(model, X_train_features, X_val_features, y_train, y_val, experiment_name, registered_model_name=None, num_kernels=1000, feature_extractor=None):
    # Set the MLflow tracking URI to the correct mlruns directory
    mlflow.set_tracking_uri("file:///Users/manasdubey2022/Desktop/NGAFID/Codebase/mlruns")
    
    mlflow.set_experiment(experiment_name=experiment_name)
    
    with mlflow.start_run(run_name=registered_model_name):
        # Log model parameters using get_params
        params = model.get_params(deep=True)
        mlflow.log_params(params)


        # if a fitted feature extractor is provided, log its parameters
        if feature_extractor:
            feature_extractor_params = feature_extractor.get_params(deep=True)
            mlflow.log_params(feature_extractor_params)
        
        # Log data split dimensions and number of kernels used
        mlflow.log_param("train_set_size", X_train_features.shape[0])
        mlflow.log_param("val_set_size", X_val_features.shape[0])
        mlflow.log_param("num_kernels", num_kernels)
        
        # Predict on the train set
        y_pred_train = model.predict(X_train_features)
        y_pred_train_binary = np.where(y_pred_train == 1, 0, 1)
        
        # Predict on the validation set
        y_pred_val = model.predict(X_val_features)
        y_pred_val_binary = np.where(y_pred_val == 1, 0, 1)
        
        # Calculate metrics for the train set
        train_class_report = classification_report(y_train, y_pred_train_binary, output_dict=True)

        # Calculate metrics for the validation set
        val_class_report = classification_report(y_val, y_pred_val_binary, output_dict=True)
        
        # Log metrics
        mlflow.log_metric("train_accuracy", train_class_report["accuracy"])
        mlflow.log_metric("val_accuracy", val_class_report["accuracy"])
        
        with open("train_classification_report.txt", "w") as f:
            f.write(classification_report(y_train, y_pred_train_binary))
        with open("val_classification_report.txt", "w") as f:
            f.write(classification_report(y_val, y_pred_val_binary))
        mlflow.log_artifact("train_classification_report.txt")
        mlflow.log_artifact("val_classification_report.txt")
        
        # Infer signature
        signature = infer_signature(X_train_features[0], y_pred_train_binary)
        
        # Optionally, log the model
        if registered_model_name:
            model_info = mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                signature=signature,
                registered_model_name=registered_model_name
            )
        
        return model_info
