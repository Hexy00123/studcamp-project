from s3_client import YandexS3Client
from fastapi import FastAPI
from dotenv import load_dotenv
from io import BytesIO
import joblib
import os 

load_dotenv()
BUCKET_NAME = 'camp-project'

client = YandexS3Client(bucket=BUCKET_NAME, yc_token=os.getenv('YC_TOKEN', None))
model_file = client.read_file('models/best/model.joblib', decode=False)
model = joblib.load(BytesIO(model_file))
print("Model is ready")

app = FastAPI()

@app.get('/')
def predict(): 
    return str(model)

if __name__ == '__main__': 
    app.run(host='0.0.0.0', port=8000)