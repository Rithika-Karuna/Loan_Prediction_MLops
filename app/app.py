from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
import pandas as pd
import numpy as np
import joblib

app = FastAPI(title="Loan Prediction API")

# Load trained model
try:
    model = joblib.load("models/model.pkl")
except Exception as e:
    print("Model loading failed:", e)
    model = None


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

    Gender (0=Female,1=Male)<br>
    <input type="number" name="Gender" required><br><br>

    Married (0=No,1=Yes)<br>
    <input type="number" name="Married" required><br><br>

    Dependents (0,1,2,3)<br>
    <input type="number" name="Dependents" required><br><br>

    Education (0=Not Graduate,1=Graduate)<br>
    <input type="number" name="Education" required><br><br>

    Self Employed (0=No,1=Yes)<br>
    <input type="number" name="Self_Employed" required><br><br>

    Property Area (0=Rural,1=Semiurban,2=Urban)<br>
    <input type="number" name="Property_Area" required><br><br>

    Applicant Income<br>
    <input type="number" name="ApplicantIncome" required><br><br>

    Coapplicant Income<br>
    <input type="number" name="CoapplicantIncome" required><br><br>

    Loan Amount<br>
    <input type="number" name="LoanAmount" required><br><br>

    Loan Amount Term<br>
    <input type="number" name="Loan_Amount_Term" required><br><br>

    Credit History (0 or 1)<br>
    <input type="number" name="Credit_History" required><br><br>

    <input type="submit" value="Predict Loan Status">

    </form>

    </body>
    </html>
    """


@app.post("/predict", response_class=HTMLResponse)
def predict(
    Gender: int = Form(...),
    Married: int = Form(...),
    Dependents: int = Form(...),
    Education: int = Form(...),
    Self_Employed: int = Form(...),
    Property_Area: int = Form(...),
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

    # Build dataframe EXACTLY like training features
    data = pd.DataFrame([{
        "Gender": Gender,
        "Married": Married,
        "Dependents": Dependents,
        "Education": Education,
        "Self_Employed": Self_Employed,
        "Loan_Amount_Term": Loan_Amount_Term,
        "Credit_History": Credit_History,
        "Property_Area": Property_Area,
        "Total_Income": Total_Income,
        "Total_Income_log": Total_Income_log,
        "LoanAmount_log": LoanAmount_log,
        "Loan_to_Income_Ratio": Loan_to_Income_Ratio
    }])

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