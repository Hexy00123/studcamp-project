from airflow.decorators import dag, task
from datetime import datetime
import sys
import os
from typing import Dict, Any, List
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor

# Import task modules
sys.path.append(os.path.join(os.path.dirname(__file__), "tasks"))
sys.path.append(os.path.join(os.path.dirname(__file__), "utils"))
from read_and_preprocess import read_and_preprocess_data
from model_train import train_model_with_gridsearch


@dag(
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["model-training"],
    description="ML pipeline for profitability prediction with modular tasks"
)
def ml_profitability_pipeline():
    
    @task
    def extract_and_preprocess() -> Dict[str, Any]:
        """Extract data from ClickHouse and preprocess it"""
        return read_and_preprocess_data(
            clickhouse_conn_id="clickhouseconn",
            test_size=0.35,
            random_state=42,
            data_limit=360_000
        )
    
    @task
    def train_model_with_grid_search(preprocessing_results: Dict[str, Any], model, params) -> Dict[str, Any]:
        """Train model with grid search using preprocessed data from S3"""
        return train_model_with_gridsearch(
            s3_folder=preprocessing_results['s3_folder'],
            model_name=model,
            param_grid=params,
            cv=3,
            scoring='neg_mean_squared_error',
        )
    
    @task
    def push_final_metrics_to_xcom(model_results: Dict[str, Any]) -> Dict[str, Any]:
        """Push final model metrics to XCom"""
        final_metrics = {
            'best_hyperparameters': model_results['best_hyperparameters'],
            'best_cv_score': model_results['best_cv_score'],
            'train_metrics': model_results['train_metrics'],
            'test_metrics': model_results['test_metrics'],
            'model_s3_path': model_results['model_info']['model_s3_path'],
            'data_s3_folder': model_results['model_info']['data_s3_folder'],
            'timestamp': model_results['model_info']['timestamp']
        }

        print("=" * 60)
        print("FINAL MODEL TRAINING RESULTS")
        print("=" * 60)
        print(f"Model S3 Path: {final_metrics['model_s3_path']}")
        print(f"Data S3 Folder: {final_metrics['data_s3_folder']}")
        print(f"Timestamp: {final_metrics['timestamp']}")
        print(f"Best CV Score: {final_metrics['best_cv_score']:.4f}")
        print(f"Best Hyperparameters: {final_metrics['best_hyperparameters']}")
        print("\nTrain Set Metrics:")
        print(f"  MAE: {final_metrics['train_metrics']['mae']:.4f}")
        print(f"  MSE: {final_metrics['train_metrics']['mse']:.4f}")
        print(f"  R2:  {final_metrics['train_metrics']['r2']:.4f}")
        print("\nTest Set Metrics:")
        print(f"  MAE: {final_metrics['test_metrics']['mae']:.4f}")
        print(f"  MSE: {final_metrics['test_metrics']['mse']:.4f}")
        print(f"  R2:  {final_metrics['test_metrics']['r2']:.4f}")
        print("=" * 60)

        return final_metrics

    @task
    def select_and_deploy_best_model(model_metrics_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        from s3_client import YandexS3Client
        BUCKET_NAME = "camp-project"
        s3_client = YandexS3Client(bucket=BUCKET_NAME)

        # Find the best model based on test R2 score
        best_model = max(model_metrics_list, key=lambda x: x['test_metrics']['r2'])
        model_path = best_model['model_s3_path']
        model_timestamp = model_path.split('/')[1]

        print(f"Best model selected: {model_path}")
        print(f"Test R2: {best_model['test_metrics']['r2']:.4f}")
        print(f"Train R2: {best_model['train_metrics']['r2']:.4f}")

        # Clear models/best directory
        try:
            response = s3_client.client.list_objects_v2(Bucket=BUCKET_NAME, Prefix='models/best/')
            if 'Contents' in response:
                objects_to_delete = [{'Key': obj['Key']} for obj in response['Contents']]
                s3_client.client.delete_objects(Bucket=BUCKET_NAME, Delete={'Objects': objects_to_delete})
        except Exception as e:
            print(f"Warning: couldn't clear best model folder: {e}")

        # Copy model.joblib
        try:
            s3_client.client.copy_object(
                CopySource={'Bucket': BUCKET_NAME, 'Key': model_path},
                Bucket=BUCKET_NAME,
                Key='models/best/model.joblib'
            )
        except Exception as e:
            print(f"Error copying model file: {e}")

        # Copy results.json
        try:
            s3_client.client.copy_object(
                CopySource={'Bucket': BUCKET_NAME, 'Key': f"models/{model_timestamp}/results.json"},
                Bucket=BUCKET_NAME,
                Key='models/best/results.json'
            )
        except Exception as e:
            print(f"Error copying results file: {e}")

        # Save deployment metadata
        deployment_metadata = {
            'deployed_at': datetime.now().isoformat(),
            'best_model_info': best_model,
        }
        s3_client.save_file('models/best/deployment_metadata.json', deployment_metadata)

        return {
            'best_model_path': 'models/best/model.joblib',
            'best_r2_score': best_model['test_metrics']['r2'],
            'deployment_metadata': deployment_metadata
        }

    # DAG flow
    preprocessing_results = extract_and_preprocess()

    rf_results = train_model_with_grid_search.override(task_id="train_rf_model")(
        preprocessing_results, model=RandomForestRegressor,
        params={'model__n_estimators': [50, 100], 'model__max_depth': [20, 35], 'model__n_jobs': [-1]}
    )
    rf_metrics = push_final_metrics_to_xcom.override(task_id="rf_metrics")(rf_results)

    gb_results = train_model_with_grid_search.override(task_id="train_gb_model")(
        preprocessing_results, model=HistGradientBoostingRegressor,
        params={'model__max_depth': [3, 4, 6]}
    )
    gb_metrics = push_final_metrics_to_xcom.override(task_id="gb_metrics")(gb_results)

    select_and_deploy_best_model.override(task_id="select_best_model")([rf_metrics, gb_metrics])


ml_profitability_pipeline()
