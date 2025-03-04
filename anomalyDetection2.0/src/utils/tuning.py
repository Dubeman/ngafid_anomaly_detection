from sklearn.model_selection import GridSearchCV
from tsai.all import ROCKET
from typing import Dict, Any
import numpy as np
from sklearn.model_selection import ParameterGrid
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest


def tune_ocsvm_params(X: np.ndarray) -> Dict[str, Any]:
    """Tune One-Class SVM hyperparameters."""
    param_grid = {
        'kernel': ['rbf'],
        'gamma': ['scale', 'auto', 0.1, 0.01],
        'nu': [0.01, 0.05, 0.1]
    }
    
    base_model = OneClassSVM()
    grid_search = GridSearchCV(
        base_model,
        param_grid,
        scoring='neg_mean_squared_error',
        cv=5,
        n_jobs=-1
    )
    grid_search.fit(X)
    return grid_search.best_params_

def tune_iforest_params(X: np.ndarray) -> Dict[str, Any]:
    """Tune Isolation Forest hyperparameters."""
    param_grid = {
        'n_estimators': [100, 200, 500],
        'contamination': [0.01, 0.05, 0.1],
        'max_samples': ['auto', 100, 500]
    }
    
    base_model = IsolationForest()
    grid_search = GridSearchCV(
        base_model,
        param_grid,
        scoring='neg_mean_squared_error',
        cv=5,
        n_jobs=-1
    )
    grid_search.fit(X)
    return grid_search.best_params_