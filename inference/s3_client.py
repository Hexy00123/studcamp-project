import json
import boto3
import botocore.config
import yandexcloud
import os

class YandexS3Client:
    def __init__(self, bucket: str, endpoint: str = "https://storage.yandexcloud.net", yc_token: str | None = None):
        self.bucket = bucket
        self.endpoint = endpoint
        self.token = yc_token or os.getenv("YC_TOKEN")
        if self.token is None:
            raise ValueError("IAM token is required. Pass it to constructor or set YC_TOKEN env variable.")
        self.client = self._init_client()

    def _provide_yc_auth(self, request, **kwargs):
        request.headers.add_header("X-YaCloud-SubjectToken", self.token)

    def _init_client(self):
        session = boto3.Session()
        session.events.register("request-created.s3.*", self._provide_yc_auth)
        return session.client(
            "s3",
            endpoint_url=self.endpoint,
            config=botocore.config.Config(
                signature_version=botocore.UNSIGNED,
                retries={"max_attempts": 5, "mode": "standard"},
            ),
        )

    def save_file(self, key: str, data: dict | str):
        body = json.dumps(data).encode("utf-8") if isinstance(data, dict) else data.encode("utf-8")
        self.client.put_object(Bucket=self.bucket, Key=key, Body=body)

    def read_file(self, key: str, decode: bool=True) -> str:
        obj = self.client.get_object(Bucket=self.bucket, Key=key)
        if decode: 
            return obj["Body"].read().decode("utf-8")
        return obj["Body"].read()

    def list_files(self, prefix: str = "") -> list[str]:
        paginator = self.client.get_paginator("list_objects_v2")
        keys = []
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                keys.append(obj["Key"])
        return keys

    def list_directory(self, prefix: str = "") -> list[str]:
        response = self.client.list_objects_v2(
            Bucket=self.bucket,
            Prefix=prefix,
            Delimiter="/" 
        )
        return [cp["Prefix"] for cp in response.get("CommonPrefixes", [])]
