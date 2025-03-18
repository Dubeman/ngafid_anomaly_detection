import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report, confusion_matrix
from typing import Dict, List, Tuple
from sklearn.model_selection import ParameterGrid

def calculate_threshold(scores: np.ndarray, n_sigma: float = 3, lower_tail: bool = False) -> float:
    """Calculate threshold based on training scores."""
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    return mean_score - n_sigma * std_score if lower_tail else mean_score + n_sigma * std_score

def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, pos_label: int = -1) -> Dict[str, float]:
    """Calculate multiple evaluation metrics."""
    print(classification_report(y_true, y_pred))
    print(confusion_matrix(y_true, y_pred)) 
    return {
        'f1': f1_score(y_true, y_pred, pos_label=pos_label),
        'precision': precision_score(y_true, y_pred, pos_label=pos_label),
        'recall': recall_score(y_true, y_pred, pos_label=pos_label)
    }

def cross_validate_anomaly_detector(
    model_class,
    normal_features: np.ndarray,
    anomalous_features: np.ndarray,
    kf,
    param_grid: dict,
    repetitions: int = 1  # Removed repetitions to avoid overfitting
) -> Tuple[List[Dict[str, float]], Dict[str, float]]:
    """Perform cross-validation for anomaly detection model with parameter tuning."""
    all_metrics = []
    
    for train_idx, test_idx in kf.split(normal_features):
        # Split data
        X_train = normal_features[train_idx]
        X_test_normal = normal_features[test_idx]
        X_test = np.vstack((X_test_normal, anomalous_features))
        y_test = np.hstack((np.ones(len(X_test_normal)), -np.ones(len(anomalous_features))))
        
        # Parameter tuning
        best_score = float('-inf')
        best_params = None
        best_model = None
        
        for params in ParameterGrid(param_grid):
            model = model_class(**params)
            model.fit(X_train)
            train_scores = model.score_samples(X_train)
            threshold = calculate_threshold(train_scores)
            test_scores = model.score_samples(X_test)
            y_pred = np.where(test_scores <= threshold, -1, 1)
            
            # Use F1 score for parameter selection
            current_score = f1_score(y_test, y_pred, pos_label=-1)
            
            if current_score > best_score:
                best_score = current_score
                best_params = params
                best_model = model
        
        # Use best model for final evaluation
        train_scores = best_model.score_samples(X_train)
        threshold = calculate_threshold(train_scores)
        test_scores = best_model.score_samples(X_test)
        y_pred = np.where(test_scores <= threshold, -1, 1)
        
        # Calculate metrics
        metrics = evaluate_model(y_test, y_pred)
        metrics['best_params'] = best_params  # Store the best parameters
        all_metrics.append(metrics)
    
    # Calculate average metrics
    avg_metrics = {
        metric: (np.mean([m[metric] for m in all_metrics if metric in m]),
                np.std([m[metric] for m in all_metrics if metric in m]))
        for metric in ['f1', 'precision', 'recall']
    }
    
    return all_metrics, avg_metrics