from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

OPTIMAL_THRESHOLD = 0.12  # cost-minimizing threshold from src/evaluate.py

SERVICE_COLUMNS = [
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

model = joblib.load(MODELS_DIR / "churn_model.joblib")
scaler = joblib.load(DATA_DIR / "scaler.joblib")
FEATURE_COLUMNS = pd.read_csv(DATA_DIR / "X_train.csv", nrows=0).columns.tolist()

app = FastAPI(title="Churn Prediction API")


class Customer(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


def preprocess(customer: Customer) -> pd.DataFrame:
    df = pd.DataFrame([customer.model_dump()])

    df["services_count"] = df[SERVICE_COLUMNS].apply(
        lambda col: ~col.isin(["No", "No phone service", "No internet service"])
    ).sum(axis=1)
    df["new_customer"] = (df["tenure"] < 6).astype(int)

    df = pd.get_dummies(df, dtype=int)
    df = df.reindex(columns=FEATURE_COLUMNS, fill_value=0)

    scaled = scaler.transform(df)
    return pd.DataFrame(scaled, columns=FEATURE_COLUMNS)


@app.post("/predict")
def predict(customer: Customer):
    X = preprocess(customer)
    churn_risk = float(model.predict_proba(X)[0, 1])
    recommendation = "לפנות ללקוח" if churn_risk >= OPTIMAL_THRESHOLD else "לא לפנות ללקוח"
    return {"churn_risk": round(churn_risk, 4), "recommendation": recommendation}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
