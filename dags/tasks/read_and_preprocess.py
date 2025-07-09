from airflow_clickhouse_plugin.hooks.clickhouse import ClickHouseHook
import pandas as pd
import polars as pl
from functools import reduce
from sklearn.model_selection import train_test_split
from datetime import datetime
import sys
import os
import io
from typing import Dict, Any

# Import the S3 client
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "utils"))
from s3_client import YandexS3Client

BUCKET_NAME = "camp-project"


def read_and_preprocess_data(
    clickhouse_conn_id: str = "clickhouseconn",
    test_size: float = 0.2,
    random_state: int = 42,
    data_limit: int = 600_000
) -> Dict[str, Any]:
    """
    Read data from ClickHouse, preprocess it, and save train/test splits to S3
    
    Args:
        clickhouse_conn_id: ClickHouse connection ID
        test_size: Proportion of data for testing
        random_state: Random state for reproducibility
        data_limit: Maximum number of rows to fetch
    
    Returns:
        Dictionary with S3 paths and dataset metadata
    """
    
    COLUMNS = [
        "tin", "reg_number", "year", "kind", "category", "activity_code_main",
        "settlement_type", "lat", "lon", "revenue", "expenditure", "employees_count",
        "region_code", "Y4170102000000", "Y4170105000000", "Y4170106020000",
        "Y4170114000000", "Y4170120000000", "Y4170203000000", "Y4170206000000",
        "Y4170210020000", "Y4170211000000", "Y4170302000000", "Y4170315000000",
        "Y4170317000000", "Y4170901000000", "Y4170902000000", "Y4170903000000",
        "Y4171002000000", "Y4171101000000", "Y4171102000000", "Y4171103000000",
        "Y4171104000000", "Y4171105020000", "Y4171106020000", "Y4171304000000",
        "Y4171337020000", "Y4171403020000", "Y4171404020000", "Y4171503000000",
        "Y4171510030000", "Y4171701010000", "Y4171701020000", "Y4171701030000",
        "Y4171701050000", "Y4171704000000", "Y4171705020000", "Y4171908000000",
        "Y4171909010000", "Y4171914010000", "avg_rev_okved_reg", "profitability"
    ]
    
    query = f"""
    SELECT
        msp.tin,
        msp.reg_number,
        msp.year,
        msp.kind,
        msp.category,
        msp.activity_code_main,
        msp.settlement_type,
        msp.lat,
        msp.lon,
        msp.revenue,
        msp.expenditure,
        msp.employees_count,
        sf.region_code,
        sf.Y4170102000000,
        sf.Y4170105000000,
        sf.Y4170106020000,
        sf.Y4170114000000,
        sf.Y4170120000000,
        sf.Y4170203000000,
        sf.Y4170206000000,
        sf.Y4170210020000,
        sf.Y4170211000000,
        sf.Y4170302000000,
        sf.Y4170315000000,
        sf.Y4170317000000,
        sf.Y4170901000000,
        sf.Y4170902000000,
        sf.Y4170903000000,
        sf.Y4171002000000,
        sf.Y4171101000000,
        sf.Y4171102000000,
        sf.Y4171103000000,
        sf.Y4171104000000,
        sf.Y4171105020000,
        sf.Y4171106020000,
        sf.Y4171304000000,
        sf.Y4171337020000,
        sf.Y4171403020000,
        sf.Y4171404020000,
        sf.Y4171503000000,
        sf.Y4171510030000,
        sf.Y4171701010000,
        sf.Y4171701020000,
        sf.Y4171701030000,
        sf.Y4171701050000,
        sf.Y4171704000000,
        sf.Y4171705020000,
        sf.Y4171908000000,
        sf.Y4171909010000,
        sf.Y4171914010000,
        acs.avg_rev_okved_reg,
        msp.profitability
    FROM db1.MSP msp
    JOIN db1.StatFeatures sf
    ON msp.year = sf.year
    AND msp.region_code = sf.region_code
    JOIN db1.ActivityCodeStats acs
    ON msp.year = acs.year
    AND msp.region_code = acs.region_code
    AND msp.activity_code_main = acs.activity_code_main
    WHERE ABS(msp.profitability) <= 0.75 
    AND msp.revenue > 1000
    AND msp.expenditure > 1000
    LIMIT {data_limit}
    """
    
    # Extract data from ClickHouse
    print("Extracting data from ClickHouse...")
    ch_hook = ClickHouseHook(clickhouse_conn_id=clickhouse_conn_id)
    data = ch_hook.execute(query)
    print(f'Retrieved {len(data)} rows from ClickHouse')
    
    # Preprocess data
    print("Preprocessing data...")
    df = pd.DataFrame(data, columns=COLUMNS)
    df = pl.from_pandas(df)
    
    # Sort by company and year
    df = df.sort(["tin", "reg_number", "year"])
    
    # Create target: profitability for next year
    df_with_target = (
        df.join(
            df.select([
                pl.col("tin"),
                pl.col("reg_number"),
                (pl.col("year") - 1).alias("year"),
                pl.col("profitability").alias("profitability_next_year"),
            ]),
            on=["tin", "reg_number", "year"],
            how="left"
        )
    )
    
    # Filter out null and infinite values
    schema = df_with_target.schema
    numeric_types = {pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
                    pl.Float32, pl.Float64}
    
    numeric_cols = [col for col, dtype in schema.items() if dtype in numeric_types]
    other_cols = [col for col in df_with_target.columns if col not in numeric_cols]
    
    numeric_filters = [pl.col(col).is_not_null() & ~pl.col(col).is_infinite() for col in numeric_cols]
    other_filters = [pl.col(col).is_not_null() for col in other_cols]
    
    combined_filter = reduce(lambda a, b: a & b, numeric_filters + other_filters)
    df_with_target = df_with_target.filter(combined_filter)
    
    # Remove ID columns
    df_with_target = df_with_target.drop(["tin", "reg_number"])
    print(f'Preprocessed data shape: {df_with_target.shape}')
    
    # Convert to pandas for sklearn
    df_final = df_with_target.to_pandas()
    
    # Prepare features and target
    X = df_final.drop("profitability_next_year", axis=1)
    y = df_final["profitability_next_year"]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"Train set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    
    # Create timestamp for unique file naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Initialize S3 client
    s3_client = YandexS3Client(bucket=BUCKET_NAME)
    
    # Save train/test splits to S3 as parquet files
    print("Saving train/test splits to S3...")
    
    # Save X_train
    parquet_buffer = io.BytesIO()
    X_train.to_parquet(parquet_buffer, index=False)
    parquet_buffer.seek(0)
    s3_client.client.put_object(
        Bucket=BUCKET_NAME, 
        Key=f"processed/{timestamp}/X_train.parquet", 
        Body=parquet_buffer.getvalue()
    )
    
    # Save X_test
    parquet_buffer = io.BytesIO()
    X_test.to_parquet(parquet_buffer, index=False)
    parquet_buffer.seek(0)
    s3_client.client.put_object(
        Bucket=BUCKET_NAME, 
        Key=f"processed/{timestamp}/X_test.parquet", 
        Body=parquet_buffer.getvalue()
    )
    
    # Save y_train
    parquet_buffer = io.BytesIO()
    y_train.to_frame().to_parquet(parquet_buffer, index=False)
    parquet_buffer.seek(0)
    s3_client.client.put_object(
        Bucket=BUCKET_NAME, 
        Key=f"processed/{timestamp}/y_train.parquet", 
        Body=parquet_buffer.getvalue()
    )
    
    # Save y_test
    parquet_buffer = io.BytesIO()
    y_test.to_frame().to_parquet(parquet_buffer, index=False)
    parquet_buffer.seek(0)
    s3_client.client.put_object(
        Bucket=BUCKET_NAME, 
        Key=f"processed/{timestamp}/y_test.parquet", 
        Body=parquet_buffer.getvalue()
    )
    
    # Save metadata
    metadata = {
        "timestamp": timestamp,
        "train_shape": X_train.shape,
        "test_shape": X_test.shape,
        "total_features": X.shape[1],
        "target_name": "profitability_next_year",
        "test_size": test_size,
        "random_state": random_state,
        "data_limit": data_limit,
        "files": {
            "X_train": f"processed/{timestamp}/X_train.parquet",
            "X_test": f"processed/{timestamp}/X_test.parquet",
            "y_train": f"processed/{timestamp}/y_train.parquet",
            "y_test": f"processed/{timestamp}/y_test.parquet"
        }
    }
    
    s3_client.save_file(f"processed/{timestamp}/metadata.json", metadata)
    
    print(f"Successfully saved train/test splits to S3 under processed/{timestamp}/")
    
    return {
        's3_folder': f"processed/{timestamp}",
        'train_shape': X_train.shape,
        'test_shape': X_test.shape,
        'total_features': X.shape[1],
        'timestamp': timestamp,
        'metadata': metadata
    }