FROM tiangolo/uvicorn-gunicorn-fastapi

RUN pip install --no-cache-dir --upgrade scikit-learn==1.6.1 pandas boto3 yandexcloud python-dotenv dill

COPY ./inference/model_train.py /
COPY ./inference/s3_client.py /
COPY ./inference/main.py /

CMD ["fastapi", "run", "/main.py"]