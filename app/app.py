from fastapi import FastAPI
import joblib
import pandas as pd
import numpy as np

app = FastAPI()

model = joblib.load("models/model.pkl")

@app.get("/")
def home():
    return {"message": "Loan Prediction API Running"}

@app.post("/predict")
def predict(data: dict):

    df = pd.DataFrame([data])

    # Feature engineering
    df["Total_Income"] = df["ApplicantIncome"] + df["CoapplicantIncome"]
    df["Total_Income_log"] = np.log(df["Total_Income"])
    df["LoanAmount_log"] = np.log(df["LoanAmount"])
    df["Loan_to_Income_Ratio"] = df["LoanAmount"] / df["Total_Income"]

    # Drop unused columns
    df = df.drop(columns=["ApplicantIncome","CoapplicantIncome","LoanAmount"])

    prediction = model.predict(df)[0]

    result = "Approved" if prediction == 1 else "Rejected"

    return {"Loan_Status": result}