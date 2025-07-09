import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
import json
from datetime import datetime
import sys
import os
import io
from typing import Dict, Any, Optional

# Import the S3 client
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "utils"))
from s3_client import YandexS3Client

BUCKET_NAME = "camp-project"


class CyclicLatLonEncoder(BaseEstimator, TransformerMixin):
    """Custom transformer for cyclic encoding of lat/lon coordinates"""
    
    def fit(self, X, y=None):
        if hasattr(X, 'columns'):
            self.feature_names_in_ = list(X.columns)
        else:
            self.feature_names_in_ = None
        return self

    def transform(self, X):
        X_ = X.copy()
        X_['lat_sin'] = np.sin(2 * np.pi * X_['lat'] / 180)
        X_['lat_cos'] = np.cos(2 * np.pi * X_['lat'] / 180)
        X_['lon_sin'] = np.sin(2 * np.pi * X_['lon'] / 360)
        X_['lon_cos'] = np.cos(2 * np.pi * X_['lon'] / 360)
        return X_


def train_model_with_gridsearch(
    s3_folder: str,
    model_name: type = RandomForestRegressor,
    param_grid: Optional[Dict[str, Any]] = None,
    cv: int = 3,
    scoring: str = 'neg_mean_squared_error',
) -> Dict[str, Any]:
    """
    Load data from S3, train model with grid search, and save results
    """
    
    s3_client = YandexS3Client(bucket=BUCKET_NAME)
    
    print(f"Loading train/test splits from S3 folder: {s3_folder}")
    
    response = s3_client.client.get_object(Bucket=BUCKET_NAME, Key=f"{s3_folder}/X_train.parquet")
    X_train = pd.read_parquet(io.BytesIO(response['Body'].read()))

    response = s3_client.client.get_object(Bucket=BUCKET_NAME, Key=f"{s3_folder}/X_test.parquet")
    X_test = pd.read_parquet(io.BytesIO(response['Body'].read()))

    response = s3_client.client.get_object(Bucket=BUCKET_NAME, Key=f"{s3_folder}/y_train.parquet")
    y_train = pd.read_parquet(io.BytesIO(response['Body'].read())).squeeze()

    response = s3_client.client.get_object(Bucket=BUCKET_NAME, Key=f"{s3_folder}/y_test.parquet")
    y_test = pd.read_parquet(io.BytesIO(response['Body'].read())).squeeze()
    
    print(f"Loaded train set shape: {X_train.shape}")
    print(f"Loaded test set shape: {X_test.shape}")
    
    categorical_features = ['kind', 'category', 'activity_code_main', 'settlement_type']
    
    base_model = model_name()
    if param_grid is None:
        raise AttributeError("You must provide a model parameters for optimisation")

    pipeline = Pipeline(steps=[
        ('cyclic_latlon', CyclicLatLonEncoder()),
        ('preprocessor', ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ],
            remainder='passthrough'
        )),
        ('model', base_model)
    ])
    
    print(f"Starting grid search with {len(param_grid)} parameter combinations...")
    print(f"Parameter grid: {param_grid}")
    
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring=scoring,
        verbose=3
    )
    
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)
    
    train_metrics = {
        'mae': mean_absolute_error(y_train, y_train_pred),
        'mse': mean_squared_error(y_train, y_train_pred),
        'r2': r2_score(y_train, y_train_pred)
    }
    
    test_metrics = {
        'mae': mean_absolute_error(y_test, y_test_pred),
        'mse': mean_squared_error(y_test, y_test_pred),
        'r2': r2_score(y_test, y_test_pred)
    }
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("Saving model to S3...")
    model_buffer = io.BytesIO()
    joblib.dump(best_model, model_buffer)
    model_buffer.seek(0)
    
    model_key = f"models/{timestamp}/model.joblib"
    s3_client.client.put_object(
        Bucket=BUCKET_NAME,
        Key=model_key,
        Body=model_buffer.getvalue()
    )
    
    results = {
        'best_hyperparameters': grid_search.best_params_,
        'best_cv_score': grid_search.best_score_,
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'model_info': {
            'model_name': model_name.__name__,
            'model_s3_path': model_key,
            'data_s3_folder': s3_folder,
            'timestamp': timestamp,
            'cv_folds': cv,
            'scoring': scoring
        },
        'grid_search_results': {
            'cv_results': {
                'mean_test_score': grid_search.cv_results_['mean_test_score'].tolist(),
                'std_test_score': grid_search.cv_results_['std_test_score'].tolist(),
                'params': grid_search.cv_results_['params']
            }
        }
    }
    
    s3_client.save_file(f"models/{timestamp}/results.json", results)
    
    print("Model training completed!")
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")
    print(f"Train metrics: MAE={train_metrics['mae']:.4f}, MSE={train_metrics['mse']:.4f}, R2={train_metrics['r2']:.4f}")
    print(f"Test metrics: MAE={test_metrics['mae']:.4f}, MSE={test_metrics['mse']:.4f}, R2={test_metrics['r2']:.4f}")
    print(f"Model saved to: {model_key}")
    
    return results


def load_model_from_s3(model_s3_path: str):
    """
    Load a trained model from S3
    """
    s3_client = YandexS3Client(bucket=BUCKET_NAME)
    
    response = s3_client.client.get_object(
        Bucket=BUCKET_NAME,
        Key=model_s3_path
    )
    
    model = joblib.load(io.BytesIO(response['Body'].read()))
    print(f"Model loaded from: {model_s3_path}")
    
    return model


def get_model_results_from_s3(results_s3_path: str) -> Dict[str, Any]:
    """
    Load model results from S3
    """
    s3_client = YandexS3Client(bucket=BUCKET_NAME)
    
    results_json = s3_client.read_file(results_s3_path)
    results = json.loads(results_json)
    
    print(f"Results loaded from: {results_s3_path}")
    
    return results
