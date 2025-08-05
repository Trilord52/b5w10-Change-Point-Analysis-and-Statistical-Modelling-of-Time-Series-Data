"""
Bayesian Change Point Detection for Brent Oil Prices

This module implements Bayesian change point detection using PyMC3
to identify structural breaks in the Brent oil price time series.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pymc as pm
import arviz as az
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class BayesianChangePointDetector:
    """
    A class to perform Bayesian change point detection on time series data.
    """
    
    def __init__(self, data=None, log_returns=None):
        """
        Initialize the change point detector.
        
        Parameters:
        -----------
        data : pd.DataFrame, optional
            Oil price data with Date index and Price column
        log_returns : pd.Series, optional
            Log returns series
        """
        self.data = data
        self.log_returns = log_returns
        self.model = None
        self.trace = None
        self.change_points = None
        
    def prepare_data(self, data, use_log_returns=True):
        """
        Prepare data for change point detection.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Oil price data with Date index and Price column
        use_log_returns : bool
            Whether to use log returns (recommended) or raw prices
            
        Returns:
        --------
        tuple
            (time_series, dates) for modeling
        """
        self.data = data
        
        if use_log_returns:
            # Calculate log returns
            self.log_returns = np.log(data['Price'] / data['Price'].shift(1))
            self.log_returns = self.log_returns.dropna()
            
            # Use log returns for modeling
            time_series = self.log_returns.values
            dates = self.log_returns.index
            
            print(f"Using log returns for change point detection")
            print(f"Log returns mean: {time_series.mean():.4f}")
            print(f"Log returns std: {time_series.std():.4f}")
        else:
            # Use raw prices
            time_series = data['Price'].values
            dates = data.index
            
            print(f"Using raw prices for change point detection")
            print(f"Price mean: ${time_series.mean():.2f}")
            print(f"Price std: ${time_series.std():.2f}")
        
        return time_series, dates
    
    def build_single_change_point_model(self, time_series, n_changepoints=1):
        """
        Build a Bayesian model with a single change point.
        
        Parameters:
        -----------
        time_series : np.array
            Time series data for modeling
        n_changepoints : int
            Number of change points to detect (default: 1)
            
        Returns:
        --------
        pymc.Model
            PyMC3 model object
        """
        n_obs = len(time_series)
        
        with pm.Model() as model:
            # Prior for change point location (uniform over all possible positions)
            tau = pm.DiscreteUniform("tau", lower=1, upper=n_obs-1)
            
            # Priors for means before and after change point
            mu_1 = pm.Normal("mu_1", mu=0, sigma=1)
            mu_2 = pm.Normal("mu_2", mu=0, sigma=1)
            
            # Prior for standard deviation
            sigma = pm.HalfNormal("sigma", sigma=1)
            
            # Create the mean function that switches at the change point
            mu = pm.math.switch(tau >= np.arange(n_obs), mu_1, mu_2)
            
            # Likelihood
            likelihood = pm.Normal("likelihood", mu=mu, sigma=sigma, observed=time_series)
            
        self.model = model
        return model
    
    def build_multiple_change_point_model(self, time_series, n_changepoints=2):
        """
        Build a Bayesian model with multiple change points.
        
        Parameters:
        -----------
        time_series : np.array
            Time series data for modeling
        n_changepoints : int
            Number of change points to detect
            
        Returns:
        --------
        pymc.Model
            PyMC3 model object
        """
        n_obs = len(time_series)
        
        with pm.Model() as model:
            # Priors for change point locations
            taus = []
            for i in range(n_changepoints):
                if i == 0:
                    tau = pm.DiscreteUniform(f"tau_{i}", lower=1, upper=n_obs-1)
                else:
                    tau = pm.DiscreteUniform(f"tau_{i}", lower=taus[i-1]+1, upper=n_obs-1)
                taus.append(tau)
            
            # Priors for means in each segment
            mus = []
            for i in range(n_changepoints + 1):
                mu = pm.Normal(f"mu_{i}", mu=0, sigma=1)
                mus.append(mu)
            
            # Prior for standard deviation
            sigma = pm.HalfNormal("sigma", sigma=1)
            
            # Create the mean function that switches at each change point
            mu_values = []
            for i in range(n_obs):
                segment = 0
                for j, tau in enumerate(taus):
                    if i >= tau:
                        segment = j + 1
                mu_values.append(mus[segment])
            
            mu = pm.math.stack(mu_values)
            
            # Likelihood
            likelihood = pm.Normal("likelihood", mu=mu, sigma=sigma, observed=time_series)
            
        self.model = model
        return model
    
    def run_mcmc(self, draws=2000, tune=1000, chains=4, return_inferencedata=True):
        """
        Run MCMC sampling for the model.
        
        Parameters:
        -----------
        draws : int
            Number of posterior draws
        tune : int
            Number of tuning steps
        chains : int
            Number of MCMC chains
        return_inferencedata : bool
            Whether to return ArviZ InferenceData object
            
        Returns:
        --------
        arviz.InferenceData or pymc.MultiTrace
            MCMC sampling results
        """
        if self.model is None:
            raise ValueError("Model must be built first. Call build_single_change_point_model() or build_multiple_change_point_model().")
        
        print(f"Running MCMC sampling with {draws} draws, {tune} tuning steps, and {chains} chains...")
        
        with self.model:
            self.trace = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                return_inferencedata=return_inferencedata,
                random_seed=42
            )
        
        print("MCMC sampling completed!")
        return self.trace
    
    def analyze_results(self, dates):
        """
        Analyze the MCMC results and extract change point information.
        
        Parameters:
        -----------
        dates : pd.DatetimeIndex
            Date index corresponding to the time series
            
        Returns:
        --------
        dict
            Dictionary containing change point analysis results
        """
        if self.trace is None:
            raise ValueError("MCMC sampling must be run first. Call run_mcmc().")
        
        # Get summary statistics
        summary = az.summary(self.trace)
        
        # Extract change point locations
        tau_vars = [var for var in summary.index if var.startswith('tau')]
        change_points = []
        
        for tau_var in tau_vars:
            tau_mean = summary.loc[tau_var, 'mean']
            tau_hdi_lower = summary.loc[tau_var, 'hdi_3%']
            tau_hdi_upper = summary.loc[tau_var, 'hdi_97%']
            
            # Convert to dates
            tau_date = dates.iloc[int(tau_mean)]
            tau_date_lower = dates.iloc[int(tau_hdi_lower)]
            tau_date_upper = dates.iloc[int(tau_hdi_upper)]
            
            change_points.append({
                'variable': tau_var,
                'date': tau_date,
                'date_lower': tau_date_lower,
                'date_upper': tau_date_upper,
                'index': int(tau_mean),
                'index_lower': int(tau_hdi_lower),
                'index_upper': int(tau_hdi_upper)
            })
        
        # Extract parameter estimates
        mu_vars = [var for var in summary.index if var.startswith('mu')]
        mus = {}
        for mu_var in mu_vars:
            mus[mu_var] = {
                'mean': summary.loc[mu_var, 'mean'],
                'std': summary.loc[mu_var, 'std'],
                'hdi_lower': summary.loc[mu_var, 'hdi_3%'],
                'hdi_upper': summary.loc[mu_var, 'hdi_97%']
            }
        
        # Extract sigma estimate
        sigma = {
            'mean': summary.loc['sigma', 'mean'],
            'std': summary.loc['sigma', 'std'],
            'hdi_lower': summary.loc['sigma', 'hdi_3%'],
            'hdi_upper': summary.loc['sigma', 'hdi_97%']
        }
        
        results = {
            'change_points': change_points,
            'mus': mus,
            'sigma': sigma,
            'summary': summary
        }
        
        self.change_points = change_points
        
        return results
    
    def plot_trace(self, figsize=(15, 10)):
        """
        Plot MCMC trace plots to check convergence.
        
        Parameters:
        -----------
        figsize : tuple
            Figure size (width, height)
        """
        if self.trace is None:
            raise ValueError("MCMC sampling must be run first. Call run_mcmc().")
        
        az.plot_trace(self.trace, figsize=figsize)
        plt.tight_layout()
        plt.show()
    
    def plot_posterior(self, figsize=(15, 10)):
        """
        Plot posterior distributions.
        
        Parameters:
        -----------
        figsize : tuple
            Figure size (width, height)
        """
        if self.trace is None:
            raise ValueError("MCMC sampling must be run first. Call run_mcmc().")
        
        az.plot_posterior(self.trace, figsize=figsize)
        plt.tight_layout()
        plt.show()
    
    def plot_change_points(self, dates, time_series, figsize=(15, 8)):
        """
        Plot the time series with detected change points.
        
        Parameters:
        -----------
        dates : pd.DatetimeIndex
            Date index
        time_series : np.array
            Time series data
        figsize : tuple
            Figure size (width, height)
        """
        if self.change_points is None:
            raise ValueError("Results must be analyzed first. Call analyze_results().")
        
        plt.figure(figsize=figsize)
        plt.plot(dates, time_series, linewidth=1, alpha=0.8, label='Time Series')
        
        # Plot change points
        colors = ['red', 'orange', 'green', 'blue', 'purple']
        for i, cp in enumerate(self.change_points):
            color = colors[i % len(colors)]
            plt.axvline(x=cp['date'], color=color, linestyle='--', alpha=0.8, 
                       label=f"Change Point {i+1}: {cp['date'].strftime('%Y-%m-%d')}")
            
            # Plot uncertainty interval
            plt.axvspan(cp['date_lower'], cp['date_upper'], alpha=0.2, color=color)
        
        plt.title('Time Series with Detected Change Points', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Value', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def print_results(self):
        """
        Print a summary of the change point detection results.
        """
        if self.change_points is None:
            raise ValueError("Results must be analyzed first. Call analyze_results().")
        
        print("=" * 60)
        print("BAYESIAN CHANGE POINT DETECTION RESULTS")
        print("=" * 60)
        
        print(f"\nDetected {len(self.change_points)} change point(s):")
        for i, cp in enumerate(self.change_points):
            print(f"\nChange Point {i+1}:")
            print(f"  Date: {cp['date'].strftime('%Y-%m-%d')}")
            print(f"  95% HDI: {cp['date_lower'].strftime('%Y-%m-%d')} to {cp['date_upper'].strftime('%Y-%m-%d')}")
            print(f"  Index: {cp['index']} (range: {cp['index_lower']} to {cp['index_upper']})")


def main():
    """
    Main function to demonstrate change point detection.
    """
    # This would be called after loading data
    print("Bayesian Change Point Detection Module")
    print("Use this module with your data after loading and preprocessing.")


if __name__ == "__main__":
    main() 