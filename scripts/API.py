import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from joblib import load
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from typing import List

# Working directory
import os, sys
sys.path.append(os.path.abspath('..'))

# Load the trained model and scaler
model_path = "../data/random_forest_fraud_model_20250128_005524.pkl"
scaler_path = "../data/scaler.pkl"
model = load(model_path)
scaler = load(scaler_path)

# Define the FastAPI app
app = FastAPI(title="Fraud Detection API")

# Define the request structure
class TransactionData(BaseModel):
    TransactionId: int
    BatchId: int
    AccountId: int
    SubscriptionId: int
    CustomerId: int
    CurrencyCode: str
    CountryCode: int
    ProviderId: int
    ProductId: int
    ProductCategory: str
    ChannelId: int
    Amount: float
    Value: float
    TransactionStartTime: str
    PricingStrategy: int

@app.post("/predict")
async def predict(transaction: TransactionData):
    try:
        # Convert input to DataFrame
        input_data = pd.DataFrame([transaction.dict()])

        # Parse datetime
        input_data['TransactionStartTime'] = pd.to_datetime(input_data['TransactionStartTime'])

        # Compute aggregate features
        aggregate_features = transaction_history.groupby('AccountId').agg(
            TotalAmount=pd.NamedAgg(column='Amount', aggfunc='sum'),
            TotalDebits=pd.NamedAgg(column='Amount', aggfunc=lambda x: x[x > 0].sum()),
            TotalCredits=pd.NamedAgg(column='Amount', aggfunc=lambda x: abs(x[x < 0].sum())),
            AvgAmount=pd.NamedAgg(column='Amount', aggfunc='mean'),
            TransactionCount=pd.NamedAgg(column='TransactionStartTime', aggfunc='count'),
            StdAmount=pd.NamedAgg(column='Amount', aggfunc='std'),
            MaxAmount=pd.NamedAgg(column='Amount', aggfunc='max')
        ).reset_index()

        # Merge aggregated features with input
        input_data = input_data.merge(aggregate_features, on='AccountId', how='left')

        # Fill missing StdAmount with 0
        input_data['StdAmount'] = input_data['StdAmount'].fillna(0)

        # Extract time-based features
        input_data['TransactionHour'] = input_data['TransactionStartTime'].dt.hour
        input_data['TransactionDay'] = input_data['TransactionStartTime'].dt.day
        input_data['TransactionMonth'] = input_data['TransactionStartTime'].dt.month
        input_data['TransactionYear'] = input_data['TransactionStartTime'].dt.year
        input_data['TransactionDayOfWeek'] = input_data['TransactionStartTime'].dt.dayofweek
        input_data['IsWeekend'] = input_data['TransactionDayOfWeek'].isin([5, 6]).astype(int)

        # Time since last transaction
        input_data['TimeSinceLastTransaction'] = input_data.groupby('AccountId')['TransactionStartTime'].diff().dt.total_seconds() / 3600
        input_data['TimeSinceLastTransaction'] = input_data['TimeSinceLastTransaction'].fillna(0)

        # Drop unnecessary columns
        input_data.drop(columns=[
            'TransactionId', 'BatchId', 'AccountId', 
            'SubscriptionId', 'CustomerId', 'TransactionStartTime', 
            'CountryCode'
        ], inplace=True)

        # One-hot encode categorical features
        input_data = pd.get_dummies(input_data, columns=['ProductCategory', 'CurrencyCode'], drop_first=True)

        # Align input features with training features
        expected_features = ['Amount', 'Value', 'TransactionHour', 'IsWeekend', 'TotalAmount', 'AvgAmount', 'TransactionCount', 'StdAmount', 'TimeSinceLastTransaction']
        for col in expected_features:
            if col not in input_data.columns:
                input_data[col] = 0

        input_data = input_data[expected_features]

        # Apply scaling
        numerical_features = ['Amount', 'Value', 'TotalAmount', 'AvgAmount', 'StdAmount', 'TimeSinceLastTransaction']
        input_data[numerical_features] = scaler.transform(input_data[numerical_features])

        # Ensure correct input format
        input_data = np.array(input_data).reshape(1, -1)

        # Perform prediction
        prediction = model.predict(input_data)
        result = "Fraudulent" if prediction[0] == 1 else "Non-Fraudulent"

        return {"Prediction": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")

@app.get("/")
async def health_check():
    return {"message": "API is running successfully"}
