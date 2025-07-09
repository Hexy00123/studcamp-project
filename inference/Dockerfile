FROM tiangolo/uvicorn-gunicorn-fastapi

# Install dependencies
RUN pip install --no-cache-dir --upgrade scikit-learn==1.6.1 pandas boto3 yandexcloud python-dotenv dill clickhouse_driver

# Copy files
COPY ./inference/model_train.py /
COPY ./inference/s3_client.py /
COPY ./inference/main.py /
COPY ./inference/entrypoint.sh /entrypoint.sh

# Make the entrypoint script executable
RUN chmod +x /entrypoint.sh

# Set the entrypoint
ENTRYPOINT ["/entrypoint.sh"]

# Command to run the application
CMD ["fastapi", "run", "/main.py"]
