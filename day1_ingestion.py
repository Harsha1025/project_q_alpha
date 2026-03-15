"""
Phase 1, Day 1: Environment & Data Ingestion
Project: Q-Alpha (Hybrid Quantum-Classical Gold Price Prediction)
Description: This script downloads 10 years of historical Gold Futures data, 
cleans it, and prepares it for classical and quantum machine learning layers.
"""

import yfinance as yf
import pandas as pd

def get_clean_gold_data(ticker="GC=F", period="10y"):
    """
    Downloads, cleans, and verifies historical financial data.
    
    Args:
        ticker (str): The Yahoo Finance ticker symbol (Default: "GC=F" for Gold Futures).
        period (str): The time period to download (Default: "10y").
        
    Returns:
        pd.DataFrame: A clean pandas DataFrame containing the historical data.
    """
    print(f"[*] Initializing data download for {ticker} over the last {period}...")
    
    # 1. Data Ingestion
    gold_data = yf.download(ticker, period=period)
    
    # Check if data was successfully downloaded
    if gold_data.empty:
        raise ValueError("No data was downloaded. Please check your internet connection or the ticker symbol.")
    
    print("[*] Download complete. Inspecting data for impurities...")
    
    # 2. Data Cleaning
    # Count initial missing values
    initial_nans = gold_data.isnull().sum().sum()
    
    # Explicitly drop rows with any missing values (NaNs)
    gold_data.dropna(inplace=True)
    
    # 3. Verification
    total_rows, total_cols = gold_data.shape
    remaining_nans = gold_data.isnull().sum().sum()
    
    print("\n--- Data Verification Summary ---")
    print(f"Original missing values found: {initial_nans}")
    print(f"Missing values after cleaning: {remaining_nans}")
    print(f"Total rows (trading days):     {total_rows}")
    print(f"Total columns (features):      {total_cols}")
    print("---------------------------------\n")
    
    print("[*] First 5 rows of the clean dataset:")
    print(gold_data.head())
    print("\n")
    
    return gold_data

if __name__ == "__main__":
    # Execute the function to verify it works as a standalone script
    df_gold = get_clean_gold_data()
    print("[*] Day 1 setup is complete and successful. Data is ready for sequencing.")
    