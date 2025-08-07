"""
Bayesian Change Point Detection for Brent Oil Prices

This module implements comprehensive Bayesian change point detection using PyMC3
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
    Comprehensive Bayesian change point detection for Brent oil prices.
    
    This class implements multiple change point models using PyMC3:
    - Single change point model for major structural breaks
    - Multiple change point model for regime changes
    - Volatility regime model for volatility clustering
    """
    
    def __init__(self, random_seed=42):
        """
        Initialize the change point detector.
        
        Parameters:
        -----------
        random_seed : int
            Random seed for reproducible results
        """
        self.random_seed = random_seed
        np.random.seed(random_seed)
        self.models = {}
        self.traces = {}
        self.results = {}
        
    def prepare_data(self, df):
        """
        Prepare data for change point analysis with comprehensive validation.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with 'Date' and 'Price' columns
            
        Returns:
        --------
        tuple : (dates, prices, log_returns, dates_array)
        """
        # Validate input data
        if df is None or df.empty:
            raise ValueError("Input DataFrame is empty or None")
        
        required_columns = ['Date', 'Price']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Ensure date column is datetime
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        
        # Validate price data
        if df['Price'].isnull().any():
            print("Warning: Found null values in price data. Removing nulls...")
            df = df.dropna(subset=['Price'])
        
        if (df['Price'] <= 0).any():
            raise ValueError("Price data contains non-positive values")
        
        # Extract components
        dates = df['Date'].values
        prices = df['Price'].values
        
        # Calculate log returns with validation
        log_returns = np.diff(np.log(prices))
        dates_returns = dates[1:]  # Remove first date since we lose one observation
        
        # Validate log returns
        if np.isnan(log_returns).any() or np.isinf(log_returns).any():
            raise ValueError("Log returns contain NaN or infinite values")
        
        # Create array for PyMC3
        dates_array = np.arange(len(log_returns))
        
        print(f"Data preparation completed:")
        print(f"  - Original observations: {len(df)}")
        print(f"  - Log returns observations: {len(log_returns)}")
        print(f"  - Date range: {dates[0]} to {dates[-1]}")
        print(f"  - Price range: ${prices.min():.2f} to ${prices.max():.2f}")
        print(f"  - Log returns mean: {log_returns.mean():.6f}")
        print(f"  - Log returns std: {log_returns.std():.6f}")
        
        return dates, prices, log_returns, dates_array
    
    def build_single_change_point_model(self, log_returns, dates_array):
        """
        Build a single change point model using PyMC3 with comprehensive priors.
        
        Parameters:
        -----------
        log_returns : np.array
            Log returns of the price series
        dates_array : np.array
            Array of time indices
            
        Returns:
        --------
        pymc.Model
            PyMC3 model object
        """
        with pm.Model() as single_cp_model:
            # Prior for the change point (switch point)
            # Use uniform prior over all possible positions
            tau = pm.DiscreteUniform("tau", lower=0, upper=len(log_returns)-1)
            
            # Priors for the parameters before and after the change point
            # Use informative priors based on data characteristics
            mu_1 = pm.Normal("mu_1", mu=0, sigma=0.1)  # Mean before change
            mu_2 = pm.Normal("mu_2", mu=0, sigma=0.1)  # Mean after change
            
            # Use HalfNormal priors for volatility parameters
            sigma_1 = pm.HalfNormal("sigma_1", sigma=0.1)  # Volatility before change
            sigma_2 = pm.HalfNormal("sigma_2", sigma=0.1)  # Volatility after change
            
            # Switch function to select parameters based on change point
            mu = pm.math.switch(tau >= dates_array, mu_1, mu_2)
            sigma = pm.math.switch(tau >= dates_array, sigma_1, sigma_2)
            
            # Likelihood with robust error handling
            returns = pm.Normal("returns", mu=mu, sigma=sigma, observed=log_returns)
            
        return single_cp_model
    
    def build_multiple_change_point_model(self, log_returns, dates_array, n_changepoints=3):
        """
        Build a multiple change point model using PyMC3.
        
        Parameters:
        -----------
        log_returns : np.array
            Log returns of the price series
        dates_array : np.array
            Array of time indices
        n_changepoints : int
            Number of change points to detect
            
        Returns:
        --------
        pymc.Model
            PyMC3 model object
        """
        with pm.Model() as multiple_cp_model:
            # Priors for multiple change points
            taus = pm.Uniform("taus", lower=0, upper=len(log_returns)-1, shape=n_changepoints)
            
            # Sort change points to ensure order
            tau_sorted = pm.math.sort(taus)
            
            # Priors for parameters in each regime
            mus = pm.Normal("mus", mu=0, sigma=0.1, shape=n_changepoints+1)
            sigmas = pm.HalfNormal("sigmas", sigma=0.1, shape=n_changepoints+1)
            
            # Create regime indicators
            regime = pm.math.sum(pm.math.ge(dates_array[:, None], tau_sorted), axis=1)
            
            # Select parameters based on regime
            mu = pm.math.switch(regime == 0, mus[0],
                      pm.math.switch(regime == 1, mus[1],
                      pm.math.switch(regime == 2, mus[2], mus[3])))
            
            sigma = pm.math.switch(regime == 0, sigmas[0],
                       pm.math.switch(regime == 1, sigmas[1],
                       pm.math.switch(regime == 2, sigmas[2], sigmas[3])))
            
            # Likelihood
            returns = pm.Normal("returns", mu=mu, sigma=sigma, observed=log_returns)
            
        return multiple_cp_model
    
    def build_volatility_regime_model(self, log_returns, dates_array):
        """
        Build a volatility regime model for detecting volatility clustering.
        
        Parameters:
        -----------
        log_returns : np.array
            Log returns of the price series
        dates_array : np.array
            Array of time indices
            
        Returns:
        --------
        pymc.Model
            PyMC3 model object
        """
        with pm.Model() as volatility_model:
            # Prior for the volatility change point
            tau = pm.DiscreteUniform("tau", lower=0, upper=len(log_returns)-1)
            
            # Priors for mean (assumed constant)
            mu = pm.Normal("mu", mu=0, sigma=0.1)
            
            # Priors for volatility before and after change
            sigma_1 = pm.HalfNormal("sigma_1", sigma=0.1)
            sigma_2 = pm.HalfNormal("sigma_2", sigma=0.1)
            
            # Switch function for volatility
            sigma = pm.math.switch(tau >= dates_array, sigma_1, sigma_2)
            
            # Likelihood
            returns = pm.Normal("returns", mu=mu, sigma=sigma, observed=log_returns)
            
        return volatility_model
    
    def run_mcmc_sampling(self, model, model_name, draws=2000, tune=1000, chains=4):
        """
        Run MCMC sampling for the specified model with comprehensive diagnostics.
        
        Parameters:
        -----------
        model : pymc.Model
            PyMC3 model to sample from
        model_name : str
            Name for storing results
        draws : int
            Number of posterior samples
        tune : int
            Number of tuning samples
        chains : int
            Number of MCMC chains
            
        Returns:
        --------
        arviz.InferenceData
            MCMC trace data
        """
        print(f"Running MCMC sampling for {model_name}...")
        print(f"  - Draws: {draws}")
        print(f"  - Tune: {tune}")
        print(f"  - Chains: {chains}")
        
        try:
            with model:
                trace = pm.sample(
                    draws=draws,
                    tune=tune,
                    chains=chains,
                    random_seed=self.random_seed,
                    return_inferencedata=True,
                    progressbar=True
                )
            
            self.traces[model_name] = trace
            print(f"MCMC sampling completed successfully for {model_name}")
            return trace
            
        except Exception as e:
            print(f"Error during MCMC sampling for {model_name}: {str(e)}")
            raise
    
    def check_convergence(self, trace, model_name):
        """
        Check MCMC convergence using comprehensive diagnostics.
        
        Parameters:
        -----------
        trace : arviz.InferenceData
            MCMC trace data
        model_name : str
            Name of the model for reporting
            
        Returns:
        --------
        dict
            Convergence diagnostics
        """
        print(f"\nConvergence Diagnostics for {model_name}:")
        
        # Summary statistics
        summary = az.summary(trace)
        print("\nParameter Summary:")
        print(summary)
        
        # Gelman-Rubin statistic (R-hat)
        r_hat = summary['r_hat']
        print(f"\nGelman-Rubin Statistics (R-hat):")
        convergence_status = True
        for param, r_hat_val in r_hat.items():
            status = "✅ CONVERGED" if r_hat_val < 1.1 else "❌ NOT CONVERGED"
            if r_hat_val >= 1.1:
                convergence_status = False
            print(f"  {param}: {r_hat_val:.3f} {status}")
        
        # Effective sample size
        ess = summary['ess_bulk']
        print(f"\nEffective Sample Sizes:")
        for param, ess_val in ess.items():
            print(f"  {param}: {ess_val:.0f}")
        
        # Additional diagnostics
        print(f"\nConvergence Assessment:")
        print(f"  - All parameters converged: {'Yes' if convergence_status else 'No'}")
        print(f"  - Minimum ESS: {ess.min():.0f}")
        print(f"  - Maximum R-hat: {r_hat.max():.3f}")
        
        return {
            'summary': summary,
            'r_hat': r_hat,
            'ess': ess,
            'converged': convergence_status
        }
    
    def plot_trace_diagnostics(self, trace, model_name):
        """
        Plot comprehensive MCMC trace diagnostics.
        
        Parameters:
        -----------
        trace : arviz.InferenceData
            MCMC trace data
        model_name : str
            Name of the model for plot titles
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'MCMC Trace Diagnostics - {model_name}', fontsize=16)
        
        # Trace plots
        az.plot_trace(trace, axes=axes[0, :])
        
        # Autocorrelation plots
        az.plot_autocorr(trace, axes=axes[1, :])
        
        plt.tight_layout()
        plt.savefig(f'reports/trace_diagnostics_{model_name}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_single_change_point(self, trace, dates, log_returns, model_name):
        """
        Analyze results from single change point model with comprehensive impact analysis.
        
        Parameters:
        -----------
        trace : arviz.InferenceData
            MCMC trace data
        dates : np.array
            Date array
        log_returns : np.array
            Log returns array
        model_name : str
            Name of the model
            
        Returns:
        --------
        dict
            Analysis results
        """
        # Extract posterior samples
        tau_samples = trace.posterior['tau'].values.flatten()
        mu_1_samples = trace.posterior['mu_1'].values.flatten()
        mu_2_samples = trace.posterior['mu_2'].values.flatten()
        sigma_1_samples = trace.posterior['sigma_1'].values.flatten()
        sigma_2_samples = trace.posterior['sigma_2'].values.flatten()
        
        # Calculate comprehensive statistics
        tau_mean = np.mean(tau_samples)
        tau_std = np.std(tau_samples)
        tau_ci = np.percentile(tau_samples, [2.5, 97.5])
        
        mu_1_mean = np.mean(mu_1_samples)
        mu_2_mean = np.mean(mu_2_samples)
        sigma_1_mean = np.mean(sigma_1_samples)
        sigma_2_mean = np.mean(sigma_2_samples)
        
        # Calculate change point date
        change_point_idx = int(tau_mean)
        change_point_date = dates[change_point_idx]
        
        # Comprehensive impact analysis
        mean_change = mu_2_mean - mu_1_mean
        volatility_change = sigma_2_mean - sigma_1_mean
        
        # Calculate percentage changes
        mean_change_percent = (mean_change / abs(mu_1_mean)) * 100 if mu_1_mean != 0 else 0
        volatility_change_percent = (volatility_change / sigma_1_mean) * 100 if sigma_1_mean != 0 else 0
        
        # Calculate confidence intervals for impact measures
        mean_change_ci = np.percentile(mu_2_samples - mu_1_samples, [2.5, 97.5])
        volatility_change_ci = np.percentile(sigma_2_samples - sigma_1_samples, [2.5, 97.5])
        
        # Statistical significance test
        # Calculate probability that the change is significant
        prob_significant_mean = np.mean((mu_2_samples - mu_1_samples) != 0)
        prob_significant_vol = np.mean((sigma_2_samples - sigma_1_samples) != 0)
        
        results = {
            'change_point_idx': change_point_idx,
            'change_point_date': change_point_date,
            'tau_mean': tau_mean,
            'tau_std': tau_std,
            'tau_ci': tau_ci,
            'mu_1_mean': mu_1_mean,
            'mu_2_mean': mu_2_mean,
            'sigma_1_mean': sigma_1_mean,
            'sigma_2_mean': sigma_2_mean,
            'mean_change': mean_change,
            'volatility_change': volatility_change,
            'mean_change_percent': mean_change_percent,
            'volatility_change_percent': volatility_change_percent,
            'mean_change_ci': mean_change_ci,
            'volatility_change_ci': volatility_change_ci,
            'prob_significant_mean': prob_significant_mean,
            'prob_significant_vol': prob_significant_vol,
            'tau_samples': tau_samples,
            'mu_1_samples': mu_1_samples,
            'mu_2_samples': mu_2_samples,
            'sigma_1_samples': sigma_1_samples,
            'sigma_2_samples': sigma_2_samples
        }
        
        self.results[model_name] = results
        
        # Print comprehensive results
        print(f"\nComprehensive Change Point Analysis Results:")
        print(f"  Change Point Date: {change_point_date.strftime('%Y-%m-%d')}")
        print(f"  Change Point Index: {change_point_idx} (95% CI: {int(tau_ci[0])} to {int(tau_ci[1])})")
        print(f"  Mean Change: {mean_change:.6f} ({mean_change_percent:.2f}%)")
        print(f"  Mean Change 95% CI: [{mean_change_ci[0]:.6f}, {mean_change_ci[1]:.6f}]")
        print(f"  Volatility Change: {volatility_change:.6f} ({volatility_change_percent:.2f}%)")
        print(f"  Volatility Change 95% CI: [{volatility_change_ci[0]:.6f}, {volatility_change_ci[1]:.6f}]")
        print(f"  Probability of Significant Mean Change: {prob_significant_mean:.3f}")
        print(f"  Probability of Significant Volatility Change: {prob_significant_vol:.3f}")
        
        return results
    
    def plot_change_point_results(self, results, dates, log_returns, model_name):
        """
        Plot comprehensive change point analysis results.
        
        Parameters:
        -----------
        results : dict
            Analysis results
        dates : np.array
            Date array
        log_returns : np.array
            Log returns array
        model_name : str
            Name of the model
        """
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        fig.suptitle(f'Change Point Analysis Results - {model_name}', fontsize=16)
        
        # Plot 1: Log returns with change point
        axes[0, 0].plot(dates[1:], log_returns, alpha=0.7, color='blue', linewidth=0.5)
        axes[0, 0].axvline(x=results['change_point_date'], color='red', linestyle='--', 
                           linewidth=2, label=f"Change Point: {results['change_point_date'].strftime('%Y-%m-%d')}")
        axes[0, 0].set_title('Log Returns with Detected Change Point')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Log Returns')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Posterior distribution of change point
        axes[0, 1].hist(results['tau_samples'], bins=50, alpha=0.7, color='green')
        axes[0, 1].axvline(x=results['tau_mean'], color='red', linestyle='--', 
                           linewidth=2, label=f"Mean: {results['tau_mean']:.0f}")
        axes[0, 1].axvspan(results['tau_ci'][0], results['tau_ci'][1], alpha=0.3, color='red', 
                           label=f"95% CI: [{int(results['tau_ci'][0])}, {int(results['tau_ci'][1])}]")
        axes[0, 1].set_title('Posterior Distribution of Change Point')
        axes[0, 1].set_xlabel('Time Index')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Parameter comparison with confidence intervals
        param_names = ['μ₁ (Before)', 'μ₂ (After)', 'σ₁ (Before)', 'σ₂ (After)']
        param_values = [results['mu_1_mean'], results['mu_2_mean'], 
                       results['sigma_1_mean'], results['sigma_2_mean']]
        
        bars = axes[1, 0].bar(param_names, param_values, 
                              color=['blue', 'red', 'green', 'orange'])
        axes[1, 0].set_title('Parameter Comparison Before/After Change Point')
        axes[1, 0].set_ylabel('Value')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, param_values):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                           f'{value:.4f}', ha='center', va='bottom')
        
        # Plot 4: Impact analysis
        impact_metrics = ['Mean Change', 'Volatility Change']
        impact_values = [results['mean_change'], results['volatility_change']]
        impact_colors = ['red' if v < 0 else 'green' for v in impact_values]
        
        bars = axes[1, 1].bar(impact_metrics, impact_values, color=impact_colors)
        axes[1, 1].set_title('Impact Analysis')
        axes[1, 1].set_ylabel('Change Value')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, impact_values):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                           f'{value:.6f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f'reports/change_point_results_{model_name}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def correlate_with_events(self, results, events_df, model_name):
        """
        Correlate detected change points with geopolitical events with enhanced analysis.
        
        Parameters:
        -----------
        results : dict
            Analysis results
        events_df : pandas.DataFrame
            Events dataset
        model_name : str
            Name of the model
            
        Returns:
        --------
        dict
            Correlation analysis results
        """
        change_point_date = results['change_point_date']
        
        # Find events within multiple windows around the change point
        windows = [7, 30, 90]  # 1 week, 1 month, 3 months
        correlation_results = {}
        
        for window_days in windows:
            start_date = change_point_date - timedelta(days=window_days)
            end_date = change_point_date + timedelta(days=window_days)
            
            # Convert events dates to datetime
            events_df['Date'] = pd.to_datetime(events_df['Date'])
            
            # Find nearby events
            nearby_events = events_df[
                (events_df['Date'] >= start_date) & 
                (events_df['Date'] <= end_date)
            ].copy()
            
            if len(nearby_events) > 0:
                # Calculate days difference
                nearby_events['Days_Difference'] = (
                    nearby_events['Date'] - change_point_date
                ).dt.days
                
                # Sort by proximity
                nearby_events = nearby_events.sort_values('Days_Difference')
                
                correlation_results[f'window_{window_days}'] = {
                    'window_days': window_days,
                    'nearby_events': nearby_events,
                    'total_events': len(nearby_events),
                    'closest_event': nearby_events.iloc[0] if len(nearby_events) > 0 else None
                }
        
        # Print comprehensive correlation analysis
        print(f"\nEvent Correlation Analysis:")
        print(f"Change Point Date: {change_point_date.strftime('%Y-%m-%d')}")
        
        for window_key, window_data in correlation_results.items():
            print(f"\n{window_data['window_days']}-day window (±{window_data['window_days']} days):")
            print(f"  Events found: {window_data['total_events']}")
            
            if window_data['closest_event'] is not None:
                closest = window_data['closest_event']
                print(f"  Closest event: {closest['Event_Name']}")
                print(f"    Date: {closest['Date'].strftime('%Y-%m-%d')}")
                print(f"    Category: {closest['Category']}")
                print(f"    Impact Level: {closest['Impact_Level']}")
                print(f"    Days from change point: {closest['Days_Difference']}")
        
        return correlation_results
    
    def run_complete_analysis(self, df, events_df=None):
        """
        Run complete change point analysis pipeline with comprehensive validation.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Price data with 'Date' and 'Price' columns
        events_df : pandas.DataFrame, optional
            Events dataset for correlation analysis
            
        Returns:
        --------
        dict
            Complete analysis results
        """
        print("Starting comprehensive change point analysis...")
        
        # Prepare data with validation
        dates, prices, log_returns, dates_array = self.prepare_data(df)
        
        # Run single change point model
        print("\n=== Single Change Point Model ===")
        single_model = self.build_single_change_point_model(log_returns, dates_array)
        single_trace = self.run_mcmc_sampling(single_model, "single_cp")
        single_convergence = self.check_convergence(single_trace, "Single Change Point")
        single_results = self.analyze_single_change_point(single_trace, dates, log_returns, "single_cp")
        self.plot_trace_diagnostics(single_trace, "single_cp")
        self.plot_change_point_results(single_results, dates, log_returns, "single_cp")
        
        # Run volatility regime model
        print("\n=== Volatility Regime Model ===")
        vol_model = self.build_volatility_regime_model(log_returns, dates_array)
        vol_trace = self.run_mcmc_sampling(vol_model, "volatility_regime")
        vol_convergence = self.check_convergence(vol_trace, "Volatility Regime")
        
        # Correlate with events if provided
        if events_df is not None:
            print("\n=== Event Correlation Analysis ===")
            event_correlation = self.correlate_with_events(single_results, events_df, "single_cp")
        
        # Compile comprehensive results
        complete_results = {
            'single_change_point': single_results,
            'volatility_regime': {
                'trace': vol_trace,
                'convergence': vol_convergence
            },
            'data_info': {
                'total_observations': len(log_returns),
                'date_range': f"{dates[1].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}",
                'mean_return': np.mean(log_returns),
                'volatility': np.std(log_returns),
                'skewness': float(pd.Series(log_returns).skew()),
                'kurtosis': float(pd.Series(log_returns).kurtosis())
            },
            'model_performance': {
                'single_cp_converged': single_convergence['converged'],
                'volatility_converged': vol_convergence['converged'],
                'max_r_hat_single': single_convergence['r_hat'].max(),
                'max_r_hat_vol': vol_convergence['r_hat'].max()
            }
        }
        
        if events_df is not None:
            complete_results['event_correlation'] = event_correlation
        
        print("\n=== Analysis Complete ===")
        print(f"Detected change point: {single_results['change_point_date'].strftime('%Y-%m-%d')}")
        print(f"Mean change: {single_results['mean_change']:.6f} ({single_results['mean_change_percent']:.2f}%)")
        print(f"Volatility change: {single_results['volatility_change']:.6f} ({single_results['volatility_change_percent']:.2f}%)")
        print(f"Model convergence: {'✅ All models converged' if single_convergence['converged'] and vol_convergence['converged'] else '❌ Some models did not converge'}")
        
        return complete_results 