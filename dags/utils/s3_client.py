import json
import boto3
import botocore.config
import yandexcloud


class YandexS3Client:
    def __init__(self, bucket: str, endpoint: str = "https://storage.yandexcloud.net"):
        self.bucket = bucket
        self.endpoint = endpoint
        self.client = self._init_client()

    def _provide_yc_auth(self, request, **kwargs):
        sdk = yandexcloud.SDK()
        token = sdk._channels._token_requester.get_token()
        request.headers.add_header("X-YaCloud-SubjectToken", token)

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
        if isinstance(data, dict):
            body = json.dumps(data).encode("utf-8")
        elif isinstance(data, str):
            body = data.encode("utf-8")
        else:
            raise TypeError("Only dict or str supported for saving.")
        self.client.put_object(Bucket=self.bucket, Key=key, Body=body)
        print(f"Saved '{key}' to bucket '{self.bucket}'.")

    def read_file(self, key: str) -> str:
        obj = self.client.get_object(Bucket=self.bucket, Key=key)
        content = obj["Body"].read().decode("utf-8")
        print(f"Read '{key}' from bucket '{self.bucket}'.")
        return content

    def list_files(self, prefix: str = "") -> list[str]:
        """List all object keys under the given prefix"""
        paginator = self.client.get_paginator("list_objects_v2")
        keys = []
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                keys.append(obj["Key"])
        print(f"Found {len(keys)} files under prefix '{prefix}'.")
        return keys

    def list_directory(self, prefix: str = "") -> list[str]:
        """List 'folders' (common prefixes) under the given prefix"""
        response = self.client.list_objects_v2(
            Bucket=self.bucket,
            Prefix=prefix,
            Delimiter="/"  # Important: this enables 'folder-like' behavior
        )
        dirs = response.get("CommonPrefixes", [])
        folder_names = [cp["Prefix"] for cp in dirs]
        print(f"Found directories under '{prefix}': {folder_names}")
        return folder_names
