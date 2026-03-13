@app.post("/predict-form", response_class=HTMLResponse)
def predict_form(
    ApplicantIncome: float = Form(...),
    CoapplicantIncome: float = Form(...),
    LoanAmount: float = Form(...),
    Credit_History: int = Form(...)
):

    total_income = ApplicantIncome + CoapplicantIncome

    df = pd.DataFrame([{
        "ApplicantIncome": ApplicantIncome,
        "CoapplicantIncome": CoapplicantIncome,
        "LoanAmount": LoanAmount,
        "Credit_History": Credit_History,
        "Total_Income": total_income,
        "Total_Income_log": np.log(total_income),
        "LoanAmount_log": np.log(LoanAmount),
        "Loan_to_Income_Ratio": LoanAmount / total_income
    }])

    prediction = model.predict(df)[0]

    result = "Approved" if prediction == 1 else "Rejected"

    return f"<h2>Loan Status: {result}</h2>"