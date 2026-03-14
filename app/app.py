from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
import pandas as pd
import numpy as np
import joblib

# Create FastAPI app
app = FastAPI(title="Loan Prediction API")

# Load trained model
try:
    model = joblib.load("models/model.pkl")
except Exception as e:
    print("Model loading failed:", e)
    model = None


# Root endpoint (opens form directly)
@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
        <head>
            <title>Loan Prediction</title>
        </head>
        <body>
            <h2>Loan Prediction Form</h2>

            <form action="/predict" method="post">

                Applicant Income:<br>
                <input type="number" name="ApplicantIncome" required><br><br>

                Coapplicant Income:<br>
                <input type="number" name="CoapplicantIncome" required><br><br>

                Loan Amount:<br>
                <input type="number" name="LoanAmount" required><br><br>

                Loan Amount Term:<br>
                <input type="number" name="Loan_Amount_Term" required><br><br>

                Credit History (0 or 1):<br>
                <input type="number" name="Credit_History" required><br><br>

                <input type="submit" value="Predict Loan Status">

            </form>

        </body>
    </html>
    """


# Prediction Endpoint
@app.post("/predict", response_class=HTMLResponse)
def predict(
    ApplicantIncome: float = Form(...),
    CoapplicantIncome: float = Form(...),
    LoanAmount: float = Form(...),
    Loan_Amount_Term: float = Form(...),
    Credit_History: float = Form(...)
):

    if model is None:
        return "<h2>Model not loaded. Check server logs.</h2>"

    # Feature Engineering
    Total_Income = ApplicantIncome + CoapplicantIncome
    Total_Income_log = np.log(Total_Income)
    LoanAmount_log = np.log(LoanAmount)
    Loan_to_Income_Ratio = LoanAmount / Total_Income

    # Create dataframe
    data = pd.DataFrame([{
        "ApplicantIncome": ApplicantIncome,
        "CoapplicantIncome": CoapplicantIncome,
        "LoanAmount": LoanAmount,
        "Loan_Amount_Term": Loan_Amount_Term,
        "Credit_History": Credit_History,
        "Total_Income": Total_Income,
        "Total_Income_log": Total_Income_log,
        "LoanAmount_log": LoanAmount_log,
        "Loan_to_Income_Ratio": Loan_to_Income_Ratio
    }])

    # Prediction
    prediction = model.predict(data)[0]

    result = "Loan Approved ✅" if prediction == 1 else "Loan Rejected ❌"

    return f"""
    <html>
        <body>

            <h2>Prediction Result</h2>
            <h3>{result}</h3>

            <br>

            <a href="/">Predict Again</a>

        </body>
    </html>
    """