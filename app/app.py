from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
import pandas as pd
import joblib

# 1️⃣ Create FastAPI app
app = FastAPI(title="Loan Prediction API")

# 2️⃣ Load trained model
model = joblib.load("models/model.pkl")


# 3️⃣ Root endpoint (health check)
@app.get("/")
def home():
    return {"message": "Loan Prediction API Running"}


# 4️⃣ HTML Form Page
@app.get("/predict-form", response_class=HTMLResponse)
def show_form():
    return """
    <html>
        <head>
            <title>Loan Prediction</title>
        </head>
        <body>
            <h2>Loan Prediction Form</h2>
            <form action="/predict-form" method="post">

                Applicant Income:<br>
                <input type="number" name="ApplicantIncome"><br><br>

                Coapplicant Income:<br>
                <input type="number" name="CoapplicantIncome"><br><br>

                Loan Amount:<br>
                <input type="number" name="LoanAmount"><br><br>

                Loan Amount Term:<br>
                <input type="number" name="Loan_Amount_Term"><br><br>

                Credit History (0 or 1):<br>
                <input type="number" name="Credit_History"><br><br>

                Total Income:<br>
                <input type="number" name="Total_Income"><br><br>

                Total Income Log:<br>
                <input type="number" step="any" name="Total_Income_log"><br><br>

                Loan Amount Log:<br>
                <input type="number" step="any" name="LoanAmount_log"><br><br>

                Loan to Income Ratio:<br>
                <input type="number" step="any" name="Loan_to_Income_Ratio"><br><br>

                <input type="submit" value="Predict Loan Status">

            </form>
        </body>
    </html>
    """


# 5️⃣ Prediction Endpoint
@app.post("/predict-form", response_class=HTMLResponse)
def predict_form(
    ApplicantIncome: float = Form(...),
    CoapplicantIncome: float = Form(...),
    LoanAmount: float = Form(...),
    Loan_Amount_Term: float = Form(...),
    Credit_History: float = Form(...),
    Total_Income: float = Form(...),
    Total_Income_log: float = Form(...),
    LoanAmount_log: float = Form(...),
    Loan_to_Income_Ratio: float = Form(...)
):

    # Convert input into dataframe
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

    # Make prediction
    prediction = model.predict(data)[0]

    result = "Loan Approved ✅" if prediction == 1 else "Loan Rejected ❌"

    return f"""
    <html>
        <body>
            <h2>Prediction Result</h2>
            <h3>{result}</h3>
            <br>
            <a href="/predict-form">Try Again</a>
        </body>
    </html>
    """