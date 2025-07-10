from s3_client import YandexS3Client
from clickhouse_driver import Client
from dotenv import load_dotenv
from fastapi import FastAPI, Query
from io import BytesIO
from sklearn.pipeline import Pipeline
import joblib
import os
import pandas as pd

load_dotenv()
BUCKET_NAME = "camp-project"
FEAT_COLUMNS = ["region_code", "year", "Y4170102000000", "Y4170105000000", "Y4170106020000", "Y4170114000000", "Y4170120000000", "Y4170203000000", "Y4170206000000", "Y4170210020000", "Y4170211000000", "Y4170302000000", "Y4170315000000", "Y4170317000000", "Y4170901000000", "Y4170902000000", "Y4170903000000", "Y4171002000000", "Y4171101000000", "Y4171102000000", "Y4171103000000", "Y4171104000000", "Y4171105020000", "Y4171106020000", "Y4171304000000", "Y4171337020000", "Y4171403020000", "Y4171404020000", "Y4171503000000", "Y4171510030000", "Y4171701010000", "Y4171701020000", "Y4171701030000", "Y4171701050000", "Y4171704000000", "Y4171705020000", "Y4171908000000", "Y4171909010000", "Y4171914010000"]
ACTIVITY_COLUMNS = ["activity", "activity_code_main", "region_code", "year", "avg_rev_okved_reg"]
MSP_COLUMNS = ["tin", "year", "reg_number", "kind", "category", "activity_code_main", "region_code", "settlement_type", "lat", "lon", "revenue", "expenditure", "employees_count", "profitability"]
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

client = YandexS3Client(bucket=BUCKET_NAME, yc_token=os.getenv("YC_TOKEN", None))
model_file = client.read_file("models/best/model.joblib", decode=False)
model: Pipeline = joblib.load(BytesIO(model_file))
print("Model is ready")

def update_local_tables():
    activity_code_stats = clickhouse_client.execute("SELECT * FROM db1.ActivityCodeStats;")
    stat_features = clickhouse_client.execute("SELECT * FROM db1.StatFeatures;")
    return activity_code_stats, stat_features

def convert_to_df(stat_features, activity_code_stats,
    tin, year, reg_number, kind, category, activity_code_main, region_code, settlement_type, lat, lon, revenue, expenditure, employees_count):
    df_stat_features = pd.DataFrame(stat_features, columns=FEAT_COLUMNS)
    df_activity_code_stats = pd.DataFrame(activity_code_stats, columns=ACTIVITY_COLUMNS)
    if year and kind and category and activity_code_main and region_code and settlement_type and lat and lon and revenue and expenditure and employees_count and tin and reg_number:
        df_msp = pd.DataFrame([(
            int(tin),
            int(year),
            int(reg_number),
            int(kind),
            int(category),
            int(activity_code_main),
            int(region_code),
            settlement_type,
            float(lat),
            float(lon),
            float(revenue),
            float(expenditure),
            int(employees_count),
            (float(revenue) - float(expenditure)) / float(revenue),
        )], columns=MSP_COLUMNS)
    else:
        df_msp = None
    return df_msp, df_stat_features, df_activity_code_stats

def merge_df(df_msp: pd.DataFrame, df_stat_features: pd.DataFrame, df_activity_code_stats: pd.DataFrame):
    df_merged = df_msp.merge(df_activity_code_stats, on=["year", "activity_code_main", "region_code"], how="left")
    df_merged = df_merged.merge(df_stat_features, on=["year", "region_code"], how="left")
    return df_merged

app = FastAPI()
activity_code_stats, stat_features = update_local_tables()

@app.get("/")
def predict(
    tin: str=Query(None),
    year: str=Query(None),
    reg_number: str=Query(None),
    kind: str=Query(None),
    category: str=Query(None),
    activity_code_main: str=Query(None),
    region_code: str=Query(None),
    settlement_type: str=Query(None),
    lat: str=Query(None),
    lon: str=Query(None),
    revenue: str=Query(None),
    expenditure: str=Query(None),
    employees_count: str=Query(None),
):
    df_msp, df_stat_features, df_activity_code_stats = convert_to_df(stat_features, activity_code_stats, tin, year, reg_number, kind, category, activity_code_main, region_code, settlement_type, lat, lon, revenue, expenditure, employees_count)
    if df_msp is not None:
        df_merged = merge_df(df_msp, df_stat_features, df_activity_code_stats)
        prediction = model.predict(df_merged)
        return f"{prediction[0]}"
    return "Query missing some values"

if __name__ == "__main__": 
    app.run(host="0.0.0.0", port=8000)
