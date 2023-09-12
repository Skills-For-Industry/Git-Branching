from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib

# Load the trained model
model_filename = "../models/logistic_regression_model.pkl"
model = joblib.load(model_filename)

# Define FastAPI app
app = FastAPI()

# Define request body using Pydantic BaseModel
class ChurnIn(BaseModel):
    age: int
    subscription_duration: int
    last_purchase: int
    average_monthly_usage: int
    customer_support_calls: int

# Endpoint for prediction
@app.post("/predict/")
def predict_churn(data: ChurnIn):
    try:
        # Extract data from request
        features = [data.age, data.subscription_duration, data.last_purchase, data.average_monthly_usage, data.customer_support_calls]
        
        # Predict
        prediction = model.predict([features])[0]
        
        return {"prediction": "Churn" if prediction == 1 else "No Churn"}
    
    except:
        raise HTTPException(status_code=400, detail="Model prediction failed.")

# Note: This code won't run here but can be used in a FastAPI deployment environment.
