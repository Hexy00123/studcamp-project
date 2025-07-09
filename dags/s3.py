from airflow.decorators import dag, task
from datetime import datetime
import json
import sys
import os

# Import the client
sys.path.append(os.path.join(os.path.dirname(__file__), "utils"))
from s3_client import YandexS3Client

BUCKET_NAME = "camp-project"

@dag(schedule=None, catchup=False, tags=["example"])
def s3_example():

    @task
    def upload():
        client = YandexS3Client(bucket=BUCKET_NAME)
        client.save_file("test/example.json", {"hello": "world", "value": 42})
        return "test/example.json"

    @task
    def list_all_files():
        client = YandexS3Client(bucket=BUCKET_NAME)
        keys = client.list_files()
        return keys

    @task
    def read(key: str):
        client = YandexS3Client(bucket=BUCKET_NAME)
        content = client.read_file(key)
        print(f"Read content: {content}")

    @task
    def list_dirs():
        client = YandexS3Client(bucket=BUCKET_NAME)
        dirs = client.list_directory()
        print(f"Directories: {dirs}")

    uploaded_key = upload()
    read(uploaded_key)
    list_all_files()
    list_dirs()

s3_example()
