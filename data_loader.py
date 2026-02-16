import pandas as pd

def load_data(path="data/housing.csv"):
    df = pd.read_csv(path)
    return df

def clean_data(df):
    # Basic cleaning
    df = df.dropna()  # simple approach for now
    return df

def explore_data(df):
    print("Shape:", df.shape)
    print("\nColumns:\n", df.columns)
    print("\nHead:\n", df.head())
    print("\nDescription:\n", df.describe())