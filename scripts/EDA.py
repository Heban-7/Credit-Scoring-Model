# Import basic libraries
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
def load_data(file_path):
    return pd.read_csv(file_path)

# DataSet Overview
def dataset_overview(data):
    print("DataSet Overview")
    print(f"\nNumber of Rows: {data.shape[0]}")
    print(f"\nNumber of Columns: {data.shape[1]}")

    print("\n\nColumn Information")
    print(data.info())


# visualize numerical feature distributions
def visualize_numerical_distributions(data):
    numerical_columns = data.select_dtypes(include=['int64', 'float64']).columns

    colors = ['blue', 'orange', 'purple', 'cyan', 'pink', 'brown', 'green', 'red', 'yellow']
    color_cycle = iter(colors)

    for col in numerical_columns:

        try:
            color = next(color_cycle)
        except StopIteration:
            color_cycle = iter(colors)
            color = next(color_cycle)
        
        plt.figure(figsize=(15, 5))
        # Histogram and KDE at the first plot
        plt.subplot(1, 2, 1)
        sns.histplot(data[col], kde=True, bins=50, color=color)
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')

        # Boxplot for outlier detection at the second plot
        plt.subplot(1, 2, 2)
        sns.boxplot(x=data[col], color=color)
        plt.title(f'Boxplot of {col}')
        
        plt.show()

# Function to plot the boxplot of the numerical features
def plot_numerical_boxplot(data, numerical_features, target):
    for feature in numerical_features:
        plt.figure(figsize=(10, 5))
        sns.boxplot(x=target, y=feature, data=data)
        plt.title('Boxplot of ' + feature + ' by ' + target)
        plt.show()

# Visualize Top User by a particular column
def top_user_plot(data, col):
    top_user = data[col].value_counts().sort_values(ascending=False)[:10]
    plt.figure(figsize=(8,6))
    top_user.plot(kind ='bar')
    plt.title(f"Top User by {col}")
    plt.xlabel(col)
    plt.ylabel('Value Count')
    plt.show()

# Top Product Type Bought by Customers
def top_product_plot(data, col):
    top_user = data[col].value_counts().sort_values(ascending=False)[:10]
    plt.figure(figsize=(8,6))
    top_user.plot(kind ='bar', color='orange')
    plt.title(f"Top Product Type Bought by Customers")
    plt.xlabel(col)
    plt.ylabel('Value Count')
    plt.show()

# Fraud Result by Provider Id
def fraud_result_by_providerid(data, column):
    plt.figure(figsize=(8, 6))
    sns.countplot(data=data, x=column, hue='FraudResult')
    plt.title('Fraud Result by Provider Id')
    plt.show()

# Category Feature Plot
def categorical_feature_plot(df):
    categorical_features = df.select_dtypes(include='object').columns
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(categorical_features):
        plt.subplot(2, 2, i+1)
        df[col].value_counts().head(10).plot(kind='bar')
        plt.title(f'Distribution of {col}')
        plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Function to plot the correlation matrix of the numerical features
def corr_matrix(df):
    plt.figure(figsize=(12, 8))
    # Select only numerical features
    numerical_df = df.select_dtypes(include=['number'])
    corr_matrix = numerical_df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title("Correlation Matrix of Numerical Features")
    plt.show()

# Function to plot the pairplot of the numerical features
def plot_pairplot(data, numerical_features):
    plt.figure(figsize=(10, 5))
    sns.pairplot(data[numerical_features])
    plt.title('Pairplot of the numerical features')
    plt.show()
