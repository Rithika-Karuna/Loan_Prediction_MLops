import pandas as pd
import mlflow
import mlflow.sklearn

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from preprocess import preprocess_data

# load data
train_df = pd.read_csv("data/loan-train.csv")

train_df = preprocess_data(train_df)

X = train_df.drop(columns=['Loan_ID','Loan_Status','ApplicantIncome','CoapplicantIncome','LoanAmount'])
y = train_df['Loan_Status'].map({'Y':1,'N':0})

X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.2,random_state=42)

# Start MLflow experiment
with mlflow.start_run():

    model = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss'
    )

    model.fit(X_train,y_train)

    preds = model.predict(X_val)

    acc = accuracy_score(y_val,preds)

    # log parameters
    mlflow.log_param("n_estimators",100)
    mlflow.log_param("max_depth",3)

    # log metric
    mlflow.log_metric("accuracy",acc)

    # log model
    mlflow.sklearn.log_model(model,"loan-model")

print("Accuracy:",acc)
import joblib
joblib.dump(model,"models/model.pkl")