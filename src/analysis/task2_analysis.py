"""
Task 2: Change Point Modeling and Insight Generation

This script implements the complete Bayesian change point detection pipeline
for analyzing Brent oil price structural breaks and correlating them with
geopolitical events.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))
from change_point_detection import BayesianChangePointDetector
from data_exploration import BrentOilDataExplorer
from event_research import OilEventResearcher

class Task2Analysis:
    """
    Comprehensive Task 2 analysis implementing Bayesian change point detection
    and correlation with geopolitical events.
    """
    
    def __init__(self):
        """Initialize the Task 2 analysis components."""
        self.data_explorer = BrentOilDataExplorer()
        self.event_researcher = OilEventResearcher()
        self.change_point_detector = BayesianChangePointDetector()
        self.results = {}
        
    def load_and_prepare_data(self):
        """
        Load and prepare all data for analysis.
        
        Returns:
        --------
        tuple : (price_data, events_data)
        """
        print("Loading and preparing data...")
        
        # Load price data
        price_data = self.data_explorer.load_data()
        print(f"Loaded {len(price_data)} price observations")
        
        # Load events data
        events_data = self.event_researcher.get_events_dataframe()
        print(f"Loaded {len(events_data)} geopolitical events")
        
        return price_data, events_data
    
    def perform_initial_eda(self, price_data):
        """
        Perform initial exploratory data analysis.
        
        Parameters:
        -----------
        price_data : pandas.DataFrame
            Price data with Date and Price columns
        """
        print("\n=== Initial Exploratory Data Analysis ===")
        
        # Basic statistics
        print("\nPrice Statistics:")
        print(f"Date Range: {price_data['Date'].min()} to {price_data['Date'].max()}")
        print(f"Price Range: ${price_data['Price'].min():.2f} to ${price_data['Price'].max():.2f}")
        print(f"Mean Price: ${price_data['Price'].mean():.2f}")
        print(f"Price Volatility: ${price_data['Price'].std():.2f}")
        
        # Calculate log returns
        price_data['Log_Returns'] = np.log(price_data['Price'] / price_data['Price'].shift(1))
        price_data = price_data.dropna()
        
        print(f"\nLog Returns Statistics:")
        print(f"Mean Return: {price_data['Log_Returns'].mean():.6f}")
        print(f"Return Volatility: {price_data['Log_Returns'].std():.6f}")
        print(f"Skewness: {price_data['Log_Returns'].skew():.3f}")
        print(f"Kurtosis: {price_data['Log_Returns'].kurtosis():.3f}")
        
        return price_data
    
    def run_bayesian_change_point_analysis(self, price_data, events_data):
        """
        Run comprehensive Bayesian change point analysis.
        
        Parameters:
        -----------
        price_data : pandas.DataFrame
            Price data with Date and Price columns
        events_data : pandas.DataFrame
            Events dataset
            
        Returns:
        --------
        dict
            Complete analysis results
        """
        print("\n=== Bayesian Change Point Analysis ===")
        
        # Run the complete analysis pipeline
        results = self.change_point_detector.run_complete_analysis(price_data, events_data)
        
        # Store results
        self.results = results
        
        return results
    
    def generate_quantitative_insights(self, results):
        """
        Generate quantitative insights from the analysis.
        
        Parameters:
        -----------
        results : dict
            Analysis results from change point detection
            
        Returns:
        --------
        dict
            Quantitative insights
        """
        print("\n=== Quantitative Impact Analysis ===")
        
        single_cp = results['single_change_point']
        
        # Calculate impact metrics
        impact_analysis = {
            'change_point_date': single_cp['change_point_date'],
            'mean_change': single_cp['mean_change'],
            'volatility_change': single_cp['volatility_change'],
            'mean_change_percent': (single_cp['mean_change'] / abs(single_cp['mu_1_mean'])) * 100 if single_cp['mu_1_mean'] != 0 else 0,
            'volatility_change_percent': (single_cp['volatility_change'] / single_cp['sigma_1_mean']) * 100 if single_cp['sigma_1_mean'] != 0 else 0,
            'before_mean': single_cp['mu_1_mean'],
            'after_mean': single_cp['mu_2_mean'],
            'before_volatility': single_cp['sigma_1_mean'],
            'after_volatility': single_cp['sigma_2_mean']
        }
        
        print(f"Change Point Date: {impact_analysis['change_point_date'].strftime('%Y-%m-%d')}")
        print(f"Mean Change: {impact_analysis['mean_change']:.6f} ({impact_analysis['mean_change_percent']:.2f}%)")
        print(f"Volatility Change: {impact_analysis['volatility_change']:.6f} ({impact_analysis['volatility_change_percent']:.2f}%)")
        print(f"Before Change - Mean: {impact_analysis['before_mean']:.6f}, Volatility: {impact_analysis['before_volatility']:.6f}")
        print(f"After Change - Mean: {impact_analysis['after_mean']:.6f}, Volatility: {impact_analysis['after_volatility']:.6f}")
        
        return impact_analysis
    
    def correlate_events_with_changes(self, results, events_data):
        """
        Correlate detected change points with geopolitical events.
        
        Parameters:
        -----------
        results : dict
            Analysis results
        events_data : pandas.DataFrame
            Events dataset
            
        Returns:
        --------
        dict
            Event correlation analysis
        """
        print("\n=== Event Correlation Analysis ===")
        
        if 'event_correlation' in results:
            correlation = results['event_correlation']
            
            print(f"Change Point: {correlation['change_point_date'].strftime('%Y-%m-%d')}")
            print(f"Events Found in ±{correlation['window_days']} day window: {correlation['total_events_in_window']}")
            
            if len(correlation['nearby_events']) > 0:
                print("\nMost Likely Associated Events:")
                for idx, event in correlation['nearby_events'].head(3).iterrows():
                    print(f"  • {event['Event_Name']} ({event['Date'].strftime('%Y-%m-%d')})")
                    print(f"    Category: {event['Category']}, Impact: {event['Impact_Level']}")
                    print(f"    Days from change point: {event['Days_Difference']}")
                    print()
            
            return correlation
        else:
            print("No event correlation data available.")
            return None
    
    def create_comprehensive_visualizations(self, price_data, results):
        """
        Create comprehensive visualizations for the analysis.
        
        Parameters:
        -----------
        price_data : pandas.DataFrame
            Price data with Date and Price columns
        results : dict
            Analysis results
        """
        print("\n=== Creating Comprehensive Visualizations ===")
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        fig.suptitle('Brent Oil Price Change Point Analysis - Comprehensive Results', fontsize=16, fontweight='bold')
        
        # Plot 1: Price series with change point
        axes[0, 0].plot(price_data['Date'], price_data['Price'], linewidth=1, alpha=0.8, color='blue')
        if 'single_change_point' in results:
            cp_date = results['single_change_point']['change_point_date']
            axes[0, 0].axvline(x=cp_date, color='red', linestyle='--', linewidth=2, 
                               label=f"Change Point: {cp_date.strftime('%Y-%m-%d')}")
        axes[0, 0].set_title('Brent Oil Price Series with Detected Change Point')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Price (USD/barrel)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Log returns with change point
        if 'Log_Returns' in price_data.columns:
            axes[0, 1].plot(price_data['Date'], price_data['Log_Returns'], linewidth=0.5, alpha=0.7, color='green')
            if 'single_change_point' in results:
                cp_date = results['single_change_point']['change_point_date']
                axes[0, 1].axvline(x=cp_date, color='red', linestyle='--', linewidth=2, 
                                   label=f"Change Point: {cp_date.strftime('%Y-%m-%d')}")
        axes[0, 1].set_title('Log Returns with Detected Change Point')
        axes[0, 1].set_xlabel('Date')
        axes[0, 1].set_ylabel('Log Returns')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Parameter comparison
        if 'single_change_point' in results:
            cp_results = results['single_change_point']
            param_names = ['μ₁ (Before)', 'μ₂ (After)', 'σ₁ (Before)', 'σ₂ (After)']
            param_values = [cp_results['mu_1_mean'], cp_results['mu_2_mean'], 
                           cp_results['sigma_1_mean'], cp_results['sigma_2_mean']]
            
            bars = axes[1, 0].bar(param_names, param_values, 
                                  color=['blue', 'red', 'green', 'orange'])
            axes[1, 0].set_title('Parameter Comparison Before/After Change Point')
            axes[1, 0].set_ylabel('Value')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Add value labels
            for bar, value in zip(bars, param_values):
                axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                               f'{value:.4f}', ha='center', va='bottom')
        
        # Plot 4: Volatility clustering
        if 'Log_Returns' in price_data.columns:
            # Calculate rolling volatility
            rolling_vol = price_data['Log_Returns'].rolling(window=30).std()
            axes[1, 1].plot(price_data['Date'], rolling_vol, linewidth=1, alpha=0.8, color='purple')
            if 'single_change_point' in results:
                cp_date = results['single_change_point']['change_point_date']
                axes[1, 1].axvline(x=cp_date, color='red', linestyle='--', linewidth=2, 
                                   label=f"Change Point: {cp_date.strftime('%Y-%m-%d')}")
        axes[1, 1].set_title('Rolling Volatility (30-day window)')
        axes[1, 1].set_xlabel('Date')
        axes[1, 1].set_ylabel('Volatility')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('reports/comprehensive_analysis_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_business_insights(self, results, impact_analysis, event_correlation):
        """
        Generate business insights and recommendations.
        
        Parameters:
        -----------
        results : dict
            Analysis results
        impact_analysis : dict
            Quantitative impact analysis
        event_correlation : dict
            Event correlation analysis
            
        Returns:
        --------
        dict
            Business insights
        """
        print("\n=== Business Insights and Recommendations ===")
        
        insights = {
            'key_findings': [],
            'investment_implications': [],
            'risk_management_recommendations': [],
            'policy_implications': []
        }
        
        # Key findings
        cp_date = impact_analysis['change_point_date']
        mean_change = impact_analysis['mean_change']
        vol_change = impact_analysis['volatility_change']
        
        insights['key_findings'].append(f"Major structural break detected on {cp_date.strftime('%Y-%m-%d')}")
        insights['key_findings'].append(f"Mean return changed by {mean_change:.6f} ({impact_analysis['mean_change_percent']:.2f}%)")
        insights['key_findings'].append(f"Volatility changed by {vol_change:.6f} ({impact_analysis['volatility_change_percent']:.2f}%)")
        
        if event_correlation and len(event_correlation['nearby_events']) > 0:
            closest_event = event_correlation['nearby_events'].iloc[0]
            insights['key_findings'].append(f"Most likely associated event: {closest_event['Event_Name']}")
        
        # Investment implications
        if mean_change > 0:
            insights['investment_implications'].append("Positive mean change suggests improved returns after the change point")
        else:
            insights['investment_implications'].append("Negative mean change suggests reduced returns after the change point")
        
        if vol_change > 0:
            insights['investment_implications'].append("Increased volatility requires enhanced risk management")
        else:
            insights['investment_implications'].append("Decreased volatility may allow for more aggressive positioning")
        
        # Risk management recommendations
        insights['risk_management_recommendations'].append("Monitor for similar geopolitical events that may trigger structural breaks")
        insights['risk_management_recommendations'].append("Adjust VaR models to account for regime changes")
        insights['risk_management_recommendations'].append("Implement dynamic hedging strategies based on volatility regimes")
        
        # Policy implications
        insights['policy_implications'].append("Geopolitical events have measurable impact on oil price dynamics")
        insights['policy_implications'].append("Policy responses should consider the persistence of structural breaks")
        insights['policy_implications'].append("Energy security policies should account for volatility regime changes")
        
        # Print insights
        print("\nKey Findings:")
        for finding in insights['key_findings']:
            print(f"  • {finding}")
        
        print("\nInvestment Implications:")
        for implication in insights['investment_implications']:
            print(f"  • {implication}")
        
        print("\nRisk Management Recommendations:")
        for rec in insights['risk_management_recommendations']:
            print(f"  • {rec}")
        
        print("\nPolicy Implications:")
        for policy in insights['policy_implications']:
            print(f"  • {policy}")
        
        return insights
    
    def run_complete_task2_analysis(self):
        """
        Run the complete Task 2 analysis pipeline.
        
        Returns:
        --------
        dict
            Complete analysis results
        """
        print("=" * 80)
        print("TASK 2: CHANGE POINT MODELING AND INSIGHT GENERATION")
        print("=" * 80)
        
        # Step 1: Load and prepare data
        price_data, events_data = self.load_and_prepare_data()
        
        # Step 2: Perform initial EDA
        price_data = self.perform_initial_eda(price_data)
        
        # Step 3: Run Bayesian change point analysis
        results = self.run_bayesian_change_point_analysis(price_data, events_data)
        
        # Step 4: Generate quantitative insights
        impact_analysis = self.generate_quantitative_insights(results)
        
        # Step 5: Correlate with events
        event_correlation = self.correlate_events_with_changes(results, events_data)
        
        # Step 6: Create visualizations
        self.create_comprehensive_visualizations(price_data, results)
        
        # Step 7: Generate business insights
        business_insights = self.generate_business_insights(results, impact_analysis, event_correlation)
        
        # Compile final results
        complete_results = {
            'price_data': price_data,
            'events_data': events_data,
            'change_point_results': results,
            'impact_analysis': impact_analysis,
            'event_correlation': event_correlation,
            'business_insights': business_insights
        }
        
        print("\n" + "=" * 80)
        print("TASK 2 ANALYSIS COMPLETE")
        print("=" * 80)
        
        return complete_results


def main():
    """
    Main function to run Task 2 analysis.
    """
    # Create analysis instance
    task2_analyzer = Task2Analysis()
    
    # Run complete analysis
    results = task2_analyzer.run_complete_task2_analysis()
    
    return results


if __name__ == "__main__":
    main()
