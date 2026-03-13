import numpy as np
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df):

    df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mode()[0])
    df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].median())
    df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0])

    df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])
    df['Married'] = df['Married'].fillna(df['Married'].mode()[0])
    df['Dependents'] = df['Dependents'].fillna(df['Dependents'].mode()[0])
    df['Self_Employed'] = df['Self_Employed'].fillna(df['Self_Employed'].mode()[0])

    df['Total_Income'] = df['ApplicantIncome'] + df['CoapplicantIncome']
    df['Total_Income_log'] = np.log(df['Total_Income'])
    df['LoanAmount_log'] = np.log(df['LoanAmount'])

    df['Loan_to_Income_Ratio'] = df['LoanAmount'] / df['Total_Income']

    le = LabelEncoder()
    cat_cols = ['Gender','Married','Education','Self_Employed','Property_Area','Dependents']

    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))

    return df