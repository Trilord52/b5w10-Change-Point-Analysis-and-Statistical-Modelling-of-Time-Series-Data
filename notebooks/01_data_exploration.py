# Task 1: Data Exploration and Foundation Setup
# Brent Oil Price Change Point Analysis

"""
This script lays the foundation for our analysis by:
1. Loading and exploring the Brent oil prices dataset
2. Understanding the data structure and characteristics
3. Performing initial statistical analysis
4. Setting up the workflow for change point detection

Dataset: Brent oil prices from May 20, 1987 to September 30, 2022
"""

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

def load_and_explore_data():
    """Load and perform initial exploration of the Brent oil prices dataset"""
    
    # Load the Brent oil prices dataset
    df = pd.read_csv('data/BrentOilPrices.csv')
    
    print("Dataset Shape:", df.shape)
    print("\nFirst few rows:")
    print(df.head())
    print("\nData types:")
    print(df.dtypes)
    print("\nMissing values:")
    print(df.isnull().sum())
    
    return df

def preprocess_data(df):
    """Preprocess the data for analysis"""
    
    # Convert Date column to datetime
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    
    # Sort by date
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Set Date as index
    df.set_index('Date', inplace=True)
    
    print("Date range:", df.index.min(), "to", df.index.max())
    print("Total number of observations:", len(df))
    print("\nFirst few rows after preprocessing:")
    print(df.head())
    
    return df

def basic_statistics(df):
    """Calculate and display basic statistics"""
    
    print("Basic Statistics:")
    print(df.describe())
    
    print("\n" + "="*50)
    print("KEY INSIGHTS:")
    print("="*50)
    print(f"Minimum price: ${df['Price'].min():.2f} per barrel")
    print(f"Maximum price: ${df['Price'].max():.2f} per barrel")
    print(f"Mean price: ${df['Price'].mean():.2f} per barrel")
    print(f"Median price: ${df['Price'].median():.2f} per barrel")
    print(f"Standard deviation: ${df['Price'].std():.2f}")
    print(f"Data spans: {(df.index.max() - df.index.min()).days} days")
    print(f"Data spans: {(df.index.max().year - df.index.min().year)} years")

def plot_time_series(df):
    """Plot the full time series"""
    
    plt.figure(figsize=(15, 8))
    plt.plot(df.index, df['Price'], linewidth=1, alpha=0.8)
    plt.title('Brent Oil Prices Over Time (1987-2022)', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price (USD per barrel)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print("\nKey periods to investigate:")
    print("1. 2008 Financial Crisis (around 2008-2009)")
    print("2. 2014-2016 Oil Price Collapse")
    print("3. 2020 COVID-19 Pandemic")
    print("4. 2022 Russia-Ukraine Conflict")

def analyze_log_returns(df):
    """Calculate and analyze log returns"""
    
    # Calculate log returns
    df['log_returns'] = np.log(df['Price'] / df['Price'].shift(1))
    
    # Remove the first row (NaN due to shift)
    df_clean = df.dropna()
    
    print("Log Returns Statistics:")
    print(df_clean['log_returns'].describe())
    
    # Plot log returns
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 1, 1)
    plt.plot(df_clean.index, df_clean['log_returns'], linewidth=0.5, alpha=0.7)
    plt.title('Log Returns Over Time', fontsize=14, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Log Returns')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    plt.hist(df_clean['log_returns'], bins=50, alpha=0.7, density=True)
    plt.title('Distribution of Log Returns', fontsize=14, fontweight='bold')
    plt.xlabel('Log Returns')
    plt.ylabel('Density')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return df_clean

def analyze_volatility(df_clean):
    """Analyze volatility patterns"""
    
    # Calculate rolling volatility (30-day window)
    df_clean['volatility_30d'] = df_clean['log_returns'].rolling(window=30).std() * np.sqrt(252)
    
    # Plot volatility
    plt.figure(figsize=(15, 8))
    plt.plot(df_clean.index, df_clean['volatility_30d'], linewidth=1, alpha=0.8)
    plt.title('30-Day Rolling Volatility of Brent Oil Prices', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Annualized Volatility', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print("\nVolatility Statistics:")
    print(df_clean['volatility_30d'].describe())

def stationarity_tests(df, df_clean):
    """Perform stationarity tests"""
    
    from statsmodels.tsa.stattools import adfuller, kpss
    
    # ADF Test for price series
    adf_result_price = adfuller(df['Price'].dropna())
    print("ADF Test for Price Series:")
    print(f"ADF Statistic: {adf_result_price[0]:.6f}")
    print(f"p-value: {adf_result_price[1]:.6f}")
    print(f"Critical values: {adf_result_price[4]}")
    
    print("\n" + "="*50)
    
    # ADF Test for log returns
    adf_result_returns = adfuller(df_clean['log_returns'])
    print("ADF Test for Log Returns:")
    print(f"ADF Statistic: {adf_result_returns[0]:.6f}")
    print(f"p-value: {adf_result_returns[1]:.6f}")
    print(f"Critical values: {adf_result_returns[4]}")
    
    print("\n" + "="*50)
    print("INTERPRETATION:")
    print("="*50)
    if adf_result_returns[1] < 0.05:
        print("✓ Log returns are stationary (suitable for change point analysis)")
    else:
        print("✗ Log returns are not stationary")
    
    if adf_result_price[1] < 0.05:
        print("✓ Price series is stationary")
    else:
        print("✗ Price series is not stationary (expected for financial time series)")

def save_results(df_clean):
    """Save processed data and display summary"""
    
    # Save processed data for further analysis
    df_clean.to_csv('data/brent_oil_processed.csv')
    print("Processed data saved to 'data/brent_oil_processed.csv'")
    
    # Display final dataset info
    print(f"\nFinal dataset shape: {df_clean.shape}")
    print(f"Date range: {df_clean.index.min()} to {df_clean.index.max()}")
    print(f"Number of observations: {len(df_clean)}")
    
    print("\n" + "="*50)
    print("SUMMARY AND NEXT STEPS:")
    print("="*50)
    print("Key Findings:")
    print("1. Data Coverage: 35+ years of Brent oil price data (1987-2022)")
    print("2. Price Range: From ~$10 to ~$140 per barrel")
    print("3. Volatility: Significant periods of high volatility")
    print("4. Stationarity: Log returns are stationary, suitable for change point analysis")
    print("\nNext Steps for Task 1:")
    print("1. Event Research: Compile structured dataset of geopolitical events")
    print("2. Assumptions Documentation: Define correlation vs. causation framework")
    print("3. Workflow Definition: Establish reproducible analysis pipeline")
    print("4. Communication Plan: Identify stakeholder communication channels")

def main():
    """Main function to run the complete data exploration"""
    
    print("="*60)
    print("TASK 1: DATA EXPLORATION AND FOUNDATION SETUP")
    print("="*60)
    
    # 1. Load and explore data
    print("\n1. Loading and exploring data...")
    df = load_and_explore_data()
    
    # 2. Preprocess data
    print("\n2. Preprocessing data...")
    df = preprocess_data(df)
    
    # 3. Basic statistics
    print("\n3. Calculating basic statistics...")
    basic_statistics(df)
    
    # 4. Time series visualization
    print("\n4. Creating time series visualization...")
    plot_time_series(df)
    
    # 5. Log returns analysis
    print("\n5. Analyzing log returns...")
    df_clean = analyze_log_returns(df)
    
    # 6. Volatility analysis
    print("\n6. Analyzing volatility...")
    analyze_volatility(df_clean)
    
    # 7. Stationarity tests
    print("\n7. Performing stationarity tests...")
    stationarity_tests(df, df_clean)
    
    # 8. Save results and summary
    print("\n8. Saving results and summary...")
    save_results(df_clean)
    
    print("\n" + "="*60)
    print("DATA EXPLORATION COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    main() 