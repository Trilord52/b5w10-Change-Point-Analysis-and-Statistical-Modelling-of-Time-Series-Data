"""
Data Exploration for Brent Oil Price Analysis

This module handles the initial exploration of the Brent oil price dataset,
including data loading, cleaning, and basic statistical analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class BrentOilDataExplorer:
    """
    A class to handle data exploration for Brent oil price analysis.
    """
    
    def __init__(self, data_path='data/BrentOilPrices.csv'):
        """
        Initialize the data explorer.
        
        Parameters:
        -----------
        data_path : str
            Path to the Brent oil prices CSV file
        """
        self.data_path = data_path
        self.data = None
        self.log_returns = None
        
    def load_data(self):
        """
        Load and preprocess the Brent oil price data.
        
        Returns:
        --------
        pd.DataFrame
            Cleaned and processed data
        """
        print("Loading Brent oil price data...")
        
        # Load the data
        self.data = pd.read_csv(self.data_path)
        
        # Convert date column to datetime
        self.data['Date'] = pd.to_datetime(self.data['Date'], format='%d-%b-%y')
        
        # Sort by date
        self.data = self.data.sort_values('Date').reset_index(drop=True)
        
        # Set date as index
        self.data.set_index('Date', inplace=True)
        
        print(f"Data loaded successfully!")
        print(f"Date range: {self.data.index.min()} to {self.data.index.max()}")
        print(f"Total observations: {len(self.data)}")
        print(f"Price range: ${self.data['Price'].min():.2f} to ${self.data['Price'].max():.2f}")
        
        return self.data
    
    def calculate_log_returns(self):
        """
        Calculate log returns for the price series.
        
        Returns:
        --------
        pd.Series
            Log returns series
        """
        if self.data is None:
            raise ValueError("Data must be loaded first. Call load_data() method.")
        
        # Calculate log returns
        self.log_returns = np.log(self.data['Price'] / self.data['Price'].shift(1))
        
        # Remove the first NaN value
        self.log_returns = self.log_returns.dropna()
        
        print(f"Log returns calculated!")
        print(f"Log returns range: {self.log_returns.min():.4f} to {self.log_returns.max():.4f}")
        print(f"Log returns mean: {self.log_returns.mean():.4f}")
        print(f"Log returns std: {self.log_returns.std():.4f}")
        
        return self.log_returns
    
    def plot_price_series(self, figsize=(15, 8)):
        """
        Plot the Brent oil price time series.
        
        Parameters:
        -----------
        figsize : tuple
            Figure size (width, height)
        """
        if self.data is None:
            raise ValueError("Data must be loaded first. Call load_data() method.")
        
        plt.figure(figsize=figsize)
        plt.plot(self.data.index, self.data['Price'], linewidth=1, alpha=0.8)
        plt.title('Brent Oil Price Time Series (1987-2022)', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Price (USD per barrel)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_log_returns(self, figsize=(15, 8)):
        """
        Plot the log returns time series.
        
        Parameters:
        -----------
        figsize : tuple
            Figure size (width, height)
        """
        if self.log_returns is None:
            raise ValueError("Log returns must be calculated first. Call calculate_log_returns() method.")
        
        plt.figure(figsize=figsize)
        plt.plot(self.log_returns.index, self.log_returns.values, linewidth=0.5, alpha=0.8)
        plt.title('Brent Oil Log Returns (1987-2022)', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Log Returns', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_price_distribution(self, figsize=(15, 5)):
        """
        Plot the distribution of prices and log returns.
        
        Parameters:
        -----------
        figsize : tuple
            Figure size (width, height)
        """
        if self.data is None or self.log_returns is None:
            raise ValueError("Data and log returns must be loaded first.")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Price distribution
        ax1.hist(self.data['Price'], bins=50, alpha=0.7, edgecolor='black')
        ax1.set_title('Distribution of Brent Oil Prices', fontweight='bold')
        ax1.set_xlabel('Price (USD per barrel)')
        ax1.set_ylabel('Frequency')
        ax1.grid(True, alpha=0.3)
        
        # Log returns distribution
        ax2.hist(self.log_returns, bins=50, alpha=0.7, edgecolor='black')
        ax2.set_title('Distribution of Log Returns', fontweight='bold')
        ax2.set_xlabel('Log Returns')
        ax2.set_ylabel('Frequency')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def get_summary_statistics(self):
        """
        Get summary statistics for the data.
        
        Returns:
        --------
        dict
            Dictionary containing summary statistics
        """
        if self.data is None:
            raise ValueError("Data must be loaded first. Call load_data() method.")
        
        stats = {
            'price_stats': self.data['Price'].describe(),
            'log_returns_stats': self.log_returns.describe() if self.log_returns is not None else None,
            'data_info': {
                'start_date': self.data.index.min(),
                'end_date': self.data.index.max(),
                'total_observations': len(self.data),
                'missing_values': self.data.isnull().sum().to_dict()
            }
        }
        
        return stats
    
    def print_summary_statistics(self):
        """
        Print summary statistics in a formatted way.
        """
        stats = self.get_summary_statistics()
        
        print("=" * 60)
        print("BRENT OIL PRICE DATA SUMMARY")
        print("=" * 60)
        
        print(f"\nData Period: {stats['data_info']['start_date'].strftime('%Y-%m-%d')} to {stats['data_info']['end_date'].strftime('%Y-%m-%d')}")
        print(f"Total Observations: {stats['data_info']['total_observations']:,}")
        print(f"Missing Values: {stats['data_info']['missing_values']}")
        
        print("\n" + "=" * 40)
        print("PRICE STATISTICS")
        print("=" * 40)
        print(stats['price_stats'])
        
        if stats['log_returns_stats'] is not None:
            print("\n" + "=" * 40)
            print("LOG RETURNS STATISTICS")
            print("=" * 40)
            print(stats['log_returns_stats'])


def main():
    """
    Main function to run the data exploration.
    """
    # Initialize the explorer
    explorer = BrentOilDataExplorer()
    
    # Load data
    data = explorer.load_data()
    
    # Calculate log returns
    log_returns = explorer.calculate_log_returns()
    
    # Print summary statistics
    explorer.print_summary_statistics()
    
    # Create visualizations
    explorer.plot_price_series()
    explorer.plot_log_returns()
    explorer.plot_price_distribution()
    
    return explorer


if __name__ == "__main__":
    main() 