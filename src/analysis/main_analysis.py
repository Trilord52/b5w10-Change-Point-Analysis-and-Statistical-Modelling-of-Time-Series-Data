"""
Main Analysis Script for Brent Oil Price Change Point Analysis

This script combines data exploration, event research, and change point detection
to provide a comprehensive analysis of how geopolitical events affect oil prices.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from data_exploration import BrentOilDataExplorer
from event_research import OilEventResearcher
import sys
import os

# Add the models directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))
from change_point_detection import BayesianChangePointDetector

class BrentOilAnalysis:
    """
    Main analysis class that combines all components of the Brent oil price analysis.
    """
    
    def __init__(self):
        """
        Initialize the main analysis class.
        """
        self.data_explorer = BrentOilDataExplorer()
        self.event_researcher = OilEventResearcher()
        self.change_point_detector = BayesianChangePointDetector()
        
        self.data = None
        self.log_returns = None
        self.events = None
        self.change_points = None
        self.analysis_results = None
        
    def run_complete_analysis(self, use_log_returns=True, n_changepoints=2):
        """
        Run the complete analysis pipeline.
        
        Parameters:
        -----------
        use_log_returns : bool
            Whether to use log returns for change point detection
        n_changepoints : int
            Number of change points to detect
            
        Returns:
        --------
        dict
            Complete analysis results
        """
        print("=" * 80)
        print("BRENT OIL PRICE CHANGE POINT ANALYSIS")
        print("=" * 80)
        
        # Step 1: Data Exploration
        print("\n" + "=" * 60)
        print("STEP 1: DATA EXPLORATION")
        print("=" * 60)
        
        self.data = self.data_explorer.load_data()
        self.log_returns = self.data_explorer.calculate_log_returns()
        self.data_explorer.print_summary_statistics()
        
        # Step 2: Event Research
        print("\n" + "=" * 60)
        print("STEP 2: EVENT RESEARCH")
        print("=" * 60)
        
        self.events = self.event_researcher.compile_major_events()
        self.event_researcher.print_events_summary()
        self.event_researcher.save_events_to_csv()
        
        # Get events within the data period
        relevant_events = self.event_researcher.get_events_for_analysis(
            self.data.index.min(), 
            self.data.index.max()
        )
        
        # Step 3: Change Point Detection
        print("\n" + "=" * 60)
        print("STEP 3: CHANGE POINT DETECTION")
        print("=" * 60)
        
        # Prepare data for change point detection
        time_series, dates = self.change_point_detector.prepare_data(
            self.data, 
            use_log_returns=use_log_returns
        )
        
        # Build model
        if n_changepoints == 1:
            self.change_point_detector.build_single_change_point_model(time_series)
        else:
            self.change_point_detector.build_multiple_change_point_model(time_series, n_changepoints)
        
        # Run MCMC sampling
        trace = self.change_point_detector.run_mcmc(draws=2000, tune=1000, chains=4)
        
        # Analyze results
        results = self.change_point_detector.analyze_results(dates)
        self.change_points = results['change_points']
        
        # Print results
        self.change_point_detector.print_results()
        
        # Step 4: Event Correlation Analysis
        print("\n" + "=" * 60)
        print("STEP 4: EVENT CORRELATION ANALYSIS")
        print("=" * 60)
        
        self.analysis_results = self.correlate_events_with_change_points(relevant_events)
        
        # Step 5: Generate Visualizations
        print("\n" + "=" * 60)
        print("STEP 5: GENERATING VISUALIZATIONS")
        print("=" * 60)
        
        self.generate_comprehensive_visualizations(dates, time_series, relevant_events)
        
        return self.analysis_results
    
    def correlate_events_with_change_points(self, events):
        """
        Correlate detected change points with geopolitical events.
        
        Parameters:
        -----------
        events : pd.DataFrame
            Relevant geopolitical events
            
        Returns:
        --------
        dict
            Correlation analysis results
        """
        if self.change_points is None:
            raise ValueError("Change points must be detected first.")
        
        correlations = []
        
        for cp in self.change_points:
            cp_date = cp['date']
            
            # Find events within a window around the change point (e.g., ±30 days)
            window_days = 30
            window_start = cp_date - pd.Timedelta(days=window_days)
            window_end = cp_date + pd.Timedelta(days=window_days)
            
            # Find events in the window
            window_events = events[
                (events['date'] >= window_start) & 
                (events['date'] <= window_end)
            ]
            
            correlation = {
                'change_point_date': cp_date,
                'change_point_index': cp['index'],
                'window_start': window_start,
                'window_end': window_end,
                'nearby_events': window_events.to_dict('records') if not window_events.empty else [],
                'num_nearby_events': len(window_events)
            }
            
            correlations.append(correlation)
        
        # Print correlation summary
        print(f"\nEvent Correlation Summary:")
        print(f"Detected {len(self.change_points)} change points")
        
        for i, corr in enumerate(correlations):
            print(f"\nChange Point {i+1} ({corr['change_point_date'].strftime('%Y-%m-%d')}):")
            print(f"  Nearby events: {corr['num_nearby_events']}")
            
            if corr['nearby_events']:
                for event in corr['nearby_events']:
                    days_diff = abs((pd.to_datetime(event['date']) - corr['change_point_date']).days)
                    print(f"    - {event['date']}: {event['event']} ({days_diff} days away)")
            else:
                print("    - No nearby events found")
        
        return {
            'correlations': correlations,
            'total_change_points': len(self.change_points),
            'total_events': len(events)
        }
    
    def generate_comprehensive_visualizations(self, dates, time_series, events):
        """
        Generate comprehensive visualizations for the analysis.
        
        Parameters:
        -----------
        dates : pd.DatetimeIndex
            Date index
        time_series : np.array
            Time series data
        events : pd.DataFrame
            Relevant events
        """
        # Create a comprehensive plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))
        
        # Plot 1: Price time series with events
        ax1.plot(self.data.index, self.data['Price'], linewidth=1, alpha=0.8)
        ax1.set_title('Brent Oil Prices with Geopolitical Events', fontweight='bold')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price (USD per barrel)')
        ax1.grid(True, alpha=0.3)
        
        # Add events to the plot
        for _, event in events.iterrows():
            ax1.axvline(x=event['date'], color='red', alpha=0.7, linestyle='--', linewidth=1)
            ax1.annotate(event['event'][:20] + '...', 
                        xy=(event['date'], ax1.get_ylim()[1]),
                        xytext=(10, 10), textcoords='offset points',
                        fontsize=8, rotation=45, ha='left')
        
        # Plot 2: Log returns with change points
        ax2.plot(dates, time_series, linewidth=0.5, alpha=0.8)
        ax2.set_title('Log Returns with Detected Change Points', fontweight='bold')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Log Returns')
        ax2.grid(True, alpha=0.3)
        
        # Add change points
        colors = ['red', 'orange', 'green', 'blue', 'purple']
        for i, cp in enumerate(self.change_points):
            color = colors[i % len(colors)]
            ax2.axvline(x=cp['date'], color=color, linestyle='--', alpha=0.8, linewidth=2)
            ax2.axvspan(cp['date_lower'], cp['date_upper'], alpha=0.2, color=color)
        
        # Plot 3: Event categories distribution
        event_categories = events['category'].value_counts()
        ax3.bar(event_categories.index, event_categories.values, alpha=0.7)
        ax3.set_title('Geopolitical Events by Category', fontweight='bold')
        ax3.set_xlabel('Event Category')
        ax3.set_ylabel('Number of Events')
        ax3.tick_params(axis='x', rotation=45)
        
        # Plot 4: Expected impact distribution
        impact_counts = events['expected_impact'].value_counts()
        colors_impact = ['green' if impact == 'Positive' else 'red' for impact in impact_counts.index]
        ax4.bar(impact_counts.index, impact_counts.values, color=colors_impact, alpha=0.7)
        ax4.set_title('Events by Expected Impact on Oil Prices', fontweight='bold')
        ax4.set_xlabel('Expected Impact')
        ax4.set_ylabel('Number of Events')
        
        plt.tight_layout()
        plt.show()
        
        # Additional plot: Timeline of events and change points
        self.plot_timeline_analysis(events)
    
    def plot_timeline_analysis(self, events):
        """
        Create a timeline visualization showing events and change points.
        
        Parameters:
        -----------
        events : pd.DataFrame
            Relevant events
        """
        fig, ax = plt.subplots(figsize=(20, 8))
        
        # Create timeline
        timeline_start = min(self.data.index.min(), events['date'].min())
        timeline_end = max(self.data.index.max(), events['date'].max())
        
        # Plot price series
        ax.plot(self.data.index, self.data['Price'], linewidth=1, alpha=0.6, color='blue', label='Oil Price')
        
        # Add events
        for _, event in events.iterrows():
            color = 'red' if event['expected_impact'] == 'Positive' else 'green'
            ax.scatter(event['date'], self.data.loc[event['date'], 'Price'] if event['date'] in self.data.index else 50,
                      color=color, s=100, alpha=0.8, zorder=5)
            ax.annotate(event['event'][:15] + '...', 
                       xy=(event['date'], self.data.loc[event['date'], 'Price'] if event['date'] in self.data.index else 50),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, rotation=45, ha='left')
        
        # Add change points
        for i, cp in enumerate(self.change_points):
            ax.axvline(x=cp['date'], color='purple', linestyle='--', alpha=0.8, linewidth=2,
                      label=f'Change Point {i+1}' if i == 0 else "")
            ax.annotate(f'CP{i+1}', 
                       xy=(cp['date'], ax.get_ylim()[1]),
                       xytext=(0, 10), textcoords='offset points',
                       fontsize=10, ha='center', fontweight='bold')
        
        ax.set_title('Timeline: Brent Oil Prices, Geopolitical Events, and Change Points', fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Price (USD per barrel)', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def save_analysis_results(self, filepath='reports/analysis_results.json'):
        """
        Save analysis results to a JSON file.
        
        Parameters:
        -----------
        filepath : str
            Path where to save the results
        """
        import json
        
        if self.analysis_results is None:
            raise ValueError("Analysis must be run first. Call run_complete_analysis().")
        
        # Prepare results for JSON serialization
        results_to_save = {
            'analysis_date': datetime.now().isoformat(),
            'data_period': {
                'start_date': self.data.index.min().isoformat(),
                'end_date': self.data.index.max().isoformat(),
                'total_observations': len(self.data)
            },
            'change_points': [
                {
                    'date': cp['date'].isoformat(),
                    'date_lower': cp['date_lower'].isoformat(),
                    'date_upper': cp['date_upper'].isoformat(),
                    'index': cp['index']
                }
                for cp in self.change_points
            ],
            'event_correlations': self.analysis_results
        }
        
        # Create reports directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(results_to_save, f, indent=2)
        
        print(f"Analysis results saved to {filepath}")


def main():
    """
    Main function to run the complete analysis.
    """
    # Initialize the analysis
    analysis = BrentOilAnalysis()
    
    # Run the complete analysis
    results = analysis.run_complete_analysis(use_log_returns=True, n_changepoints=2)
    
    # Save results
    analysis.save_analysis_results()
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    
    return analysis


if __name__ == "__main__":
    main() 