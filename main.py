import uvicorn
from fastapi import FastAPI,Request
from pydantic import BaseModel
from mlutils import predict, retrain
from typing import List
from datetime import datetime
from train import train_model,load_model
import numpy as np
import json
from fastapi import staticfiles
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from starlette.applications import Starlette
import os

# defining the main app
app = FastAPI(title="Credit risk Predictor", docs_url="/")

# calling the load_model during startup.
# this will train the model and keep it loaded for prediction.
app.add_event_handler("startup", load_model)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory=os.path.abspath(os.path.expanduser('templates')))



with open('data/columns.json') as f:
    columns=list(json.load(f)['0'].values())
query_in=dict()
for col in columns:
    query_in[col]=float
"""
categorical_features=['Status_of_existing_checking_account', 'Credit_history', 'Purpose',
       'Savings_accountbonds', 'Present_employment_since',
       'Personal_status_and_sex', 'Other_debtors__guarantors', 'Property',
       'Other_installment_plans', 'Housing', 'Job', 'Telephone',
       'foreign_worker']
{
  "Status_of_existing_checking_account": '<0 DM',
  "Duration_in_month": 6,
  "Credit_history": 'critical account',
  "Purpose": 'radio/television',
  "Credit_amount": 1169,
  "Savings_accountbonds": 'no savings account',
  "Present_employment_since": '>=7 years',
  "Installment_rate_in_percentage_of_disposable_income": 4,
  "Personal_status_and_sex": 'male:single',
  "Other_debtors__guarantors": 'none',
  "Present_residence_since": 4,
  "Property": 'real estate',
  "Age_in_years": 67,
  "Other_installment_plans": 'none',
  "Housing": 'own',
  "Number_of_existing_credits_at_this_bank": 2,
  "Job": 'skilled employee / official',
  "Number_of_people_being_liable_to_provide_maintenance_for": 1,
  "Telephone": 'yes',
  "foreign_worker": 'yes'
}
"""
# class which is expected in the payload
class QueryIn(BaseModel):
    #sepal_length: float
    #sepal_width: float
    #petal_length: float
    #petal_width: float
    #query_in
    Status_of_existing_checking_account: str
    Duration_in_month: float
    Credit_history: str
    Purpose: str
    Credit_amount: float
    Savings_accountbonds: str
    Present_employment_since: str
    Installment_rate_in_percentage_of_disposable_income: float
    Personal_status_and_sex: str
    Other_debtors__guarantors: str
    Present_residence_since: float
    Property: str
    Age_in_years: float
    Other_installment_plans: str
    Housing: str
    Number_of_existing_credits_at_this_bank: float
    Job: str
    Number_of_people_being_liable_to_provide_maintenance_for: float
    Telephone: str
    foreign_worker: str


# class which is returned in the response
class QueryOut(BaseModel):
    credit_risk_rating: str
    explanation:str

# class which is expected in the payload while re-training
class FeedbackIn(BaseModel):
    #sepal_length: float
    #sepal_width: float
    #petal_length: float
    #petal_width: float
    #query_in
    Status_of_existing_checking_account: str
    Duration_in_month: float
    Credit_history: str
    Purpose: str
    Credit_amount: float
    Savings_accountbonds: str
    Present_employment_since: str
    Installment_rate_in_percentage_of_disposable_income: float
    Personal_status_and_sex: str
    Other_debtors__guarantors: str
    Present_residence_since: float
    Property: str
    Age_in_years: float
    Other_installment_plans: str
    Housing: str
    Number_of_existing_credits_at_this_bank: float
    Job: str
    Number_of_people_being_liable_to_provide_maintenance_for: float
    Telephone: str
    foreign_worker: str

# Route definitions
@app.get("/ping")
# Healthcheck route to ensure that the API is up and running
def ping():
    return {"ping": "pong"}

@app.get("/index",response_class=HTMLResponse)
#home page for html
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/train")
def trainmodel():
    train_model()


@app.post("/predict_risk", response_model=QueryOut, status_code=200)
# Route to do the prediction using the ML model defined.
# Payload: QueryIn containing the parameters
# Response: QueryOut containing the credit_risk_rating predicted (200)
def predict_risk(query_data: QueryIn):
    output = {"credit_risk_rating": predict(query_data),"explanation": "explanation here"}
    print(output)
    return output

@app.post("/feedback_loop", status_code=200)
# Route to further train the model based on user input in form of feedback loop
# Payload: FeedbackIn containing the parameters and correct flower class
# Response: Dict with detail confirming success (200)
def feedback_loop(data: FeedbackIn):
    retrain(data)
    return {"detail": "Feedback loop successful"}


# Main function to start the app when main.py is called
if __name__ == "__main__":
    # Uvicorn is used to run the server and listen for incoming API requests on 0.0.0.0:8888
    #uvicorn.run("main:app", host="0.0.0.0", port=8888, reload=True)
    uvicorn.run("main:app", host="127.0.0.1", port=5000, reload=True) #changed the host and port
