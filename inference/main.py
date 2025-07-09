from s3_client import YandexS3Client
from clickhouse_driver import Client
from dotenv import load_dotenv
from fastapi import FastAPI, Query
from io import BytesIO
import joblib
import os 

load_dotenv()
BUCKET_NAME = 'camp-project'
host = os.getenv("CLICKHOUSE_HOST")
user = os.getenv("CLICKHOUSE_USER")
password = os.getenv("CLICKHOUSE_PASSWORD")
port = int(os.getenv("CLICKHOUSE_PORT", 9440))
clickhouse_client = Client(
    host=host,
    user=user,
    password=password,
    port=port,
    secure=True,
    verify=True,
    ca_certs="/usr/local/share/ca-certificates/Yandex/RootCA.crt",
)

client = YandexS3Client(bucket=BUCKET_NAME, yc_token=os.getenv('YC_TOKEN', None))
model_file = client.read_file('models/best/model.joblib', decode=False)
model = joblib.load(BytesIO(model_file))
print("Model is ready")

app = FastAPI()

@app.get('/')
def predict(): 
    return str(model)

@app.get('/click')
def clickhouse(query: str = Query(None)):
    if query:
        return clickhouse_client.execute(query)
    return clickhouse_client.execute("SELECT version()")

if __name__ == '__main__': 
    app.run(host='0.0.0.0', port=8000)