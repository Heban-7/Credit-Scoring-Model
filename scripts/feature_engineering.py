import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 

# Load the data
def load_data(file_path):
    data = pd.read_csv(file_path, parse_dates=['TransactionStartTime'])
    return data

# Compute aggregate features
def aggregate_features(df):
    aggregate_features = df.groupby('AccountId').agg(
        TotalAmount=pd.NamedAgg(column='Amount', aggfunc='sum'),
        TotalDebits=pd.NamedAgg(column='Amount', aggfunc=lambda x: x[x > 0].sum()),
        TotalCredits=pd.NamedAgg(column='Amount', aggfunc=lambda x: abs(x[x < 0].sum())),
        AvgAmount=pd.NamedAgg(column='Amount', aggfunc='mean'),
        TransactionCount=pd.NamedAgg(column='TransactionId', aggfunc='count'),
        StdAmount=pd.NamedAgg(column='Amount', aggfunc='std'),
        MaxAmount=pd.NamedAgg(column='Amount', aggfunc='max')
    ).reset_index()
    return aggregate_features

# Compute time-based features
def extract_time_features(df):
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    df['TransactionHour'] = df['TransactionStartTime'].dt.hour
    df['TransactionDay'] = df['TransactionStartTime'].dt.day
    df['TransactionMonth'] = df['TransactionStartTime'].dt.month
    df['TransactionYear'] = df['TransactionStartTime'].dt.year
    # Day of the week (0=Monday, 6=Sunday)
    df['TransactionDayOfWeek'] = df['TransactionStartTime'].dt.dayofweek

    # Weekend flag
    df['IsWeekend'] = df['TransactionDayOfWeek'].isin([5, 6]).astype(int)

    # Time since last transaction (for behavioral analysis)
    df['TimeSinceLastTransaction'] = df.groupby('AccountId')['TransactionStartTime'].diff().dt.total_seconds() / 3600  # Hours
    
    df['TimeSinceLastTransaction'] = df['TimeSinceLastTransaction'].fillna(0)
    
    return df