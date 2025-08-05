"""
Event Research for Brent Oil Price Analysis

This module handles the research and compilation of major geopolitical events
that have affected Brent oil prices over the past decades.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class OilEventResearcher:
    """
    A class to handle research and compilation of oil-related geopolitical events.
    """
    
    def __init__(self):
        """
        Initialize the event researcher.
        """
        self.events = None
        
    def compile_major_events(self):
        """
        Compile a list of major geopolitical events that affected oil prices.
        
        Returns:
        --------
        pd.DataFrame
            DataFrame containing event information
        """
        # Major geopolitical events affecting oil prices (1987-2022)
        events_data = [
            # 1987-1999: Early period
            {
                'date': '1987-10-19',
                'event': 'Black Monday Stock Market Crash',
                'category': 'Financial Crisis',
                'description': 'Global stock market crash affecting commodity prices',
                'expected_impact': 'Negative',
                'price_change': 'Decrease'
            },
            {
                'date': '1990-08-02',
                'event': 'Iraq Invasion of Kuwait',
                'category': 'Military Conflict',
                'description': 'Iraq invades Kuwait, leading to Gulf War',
                'expected_impact': 'Positive',
                'price_change': 'Increase'
            },
            {
                'date': '1991-01-17',
                'event': 'Gulf War Begins',
                'category': 'Military Conflict',
                'description': 'Operation Desert Storm begins',
                'expected_impact': 'Positive',
                'price_change': 'Increase'
            },
            
            # 2000-2009: 2000s period
            {
                'date': '2001-09-11',
                'event': '9/11 Terrorist Attacks',
                'category': 'Terrorism',
                'description': 'Terrorist attacks on World Trade Center',
                'expected_impact': 'Positive',
                'price_change': 'Increase'
            },
            {
                'date': '2003-03-20',
                'event': 'Iraq War Begins',
                'category': 'Military Conflict',
                'description': 'US-led invasion of Iraq',
                'expected_impact': 'Positive',
                'price_change': 'Increase'
            },
            {
                'date': '2008-09-15',
                'event': 'Lehman Brothers Bankruptcy',
                'category': 'Financial Crisis',
                'description': 'Global financial crisis begins',
                'expected_impact': 'Negative',
                'price_change': 'Decrease'
            },
            
            # 2010-2019: 2010s period
            {
                'date': '2011-02-17',
                'event': 'Arab Spring Begins',
                'category': 'Political Unrest',
                'description': 'Political unrest in Middle East and North Africa',
                'expected_impact': 'Positive',
                'price_change': 'Increase'
            },
            {
                'date': '2011-03-19',
                'event': 'Libya Civil War',
                'category': 'Military Conflict',
                'description': 'Civil war in Libya affecting oil production',
                'expected_impact': 'Positive',
                'price_change': 'Increase'
            },
            {
                'date': '2014-06-20',
                'event': 'ISIS Seizes Mosul',
                'category': 'Military Conflict',
                'description': 'ISIS gains control of major Iraqi city',
                'expected_impact': 'Positive',
                'price_change': 'Increase'
            },
            {
                'date': '2014-11-27',
                'event': 'OPEC Decision Not to Cut Production',
                'category': 'OPEC Policy',
                'description': 'OPEC maintains production levels despite oversupply',
                'expected_impact': 'Negative',
                'price_change': 'Decrease'
            },
            
            # 2020-2022: Recent period
            {
                'date': '2020-01-23',
                'event': 'COVID-19 Pandemic Begins',
                'category': 'Global Crisis',
                'description': 'COVID-19 declared global pandemic',
                'expected_impact': 'Negative',
                'price_change': 'Decrease'
            },
            {
                'date': '2020-04-20',
                'event': 'Oil Price Goes Negative',
                'category': 'Market Crisis',
                'description': 'WTI crude oil futures go negative for first time',
                'expected_impact': 'Negative',
                'price_change': 'Decrease'
            },
            {
                'date': '2021-11-04',
                'event': 'OPEC+ Agrees to Gradual Production Increase',
                'category': 'OPEC Policy',
                'description': 'OPEC+ agrees to increase production by 400k bpd',
                'expected_impact': 'Negative',
                'price_change': 'Decrease'
            },
            {
                'date': '2022-02-24',
                'event': 'Russia Invades Ukraine',
                'category': 'Military Conflict',
                'description': 'Russian invasion of Ukraine begins',
                'expected_impact': 'Positive',
                'price_change': 'Increase'
            },
            {
                'date': '2022-06-02',
                'event': 'EU Agrees to Russian Oil Ban',
                'category': 'Economic Sanctions',
                'description': 'EU agrees to ban 90% of Russian oil imports',
                'expected_impact': 'Positive',
                'price_change': 'Increase'
            }
        ]
        
        # Create DataFrame
        self.events = pd.DataFrame(events_data)
        
        # Convert date to datetime
        self.events['date'] = pd.to_datetime(self.events['date'])
        
        # Sort by date
        self.events = self.events.sort_values('date').reset_index(drop=True)
        
        print(f"Compiled {len(self.events)} major geopolitical events affecting oil prices")
        print(f"Date range: {self.events['date'].min().strftime('%Y-%m-%d')} to {self.events['date'].max().strftime('%Y-%m-%d')}")
        
        return self.events
    
    def save_events_to_csv(self, filepath='data/geopolitical_events.csv'):
        """
        Save the events DataFrame to a CSV file.
        
        Parameters:
        -----------
        filepath : str
            Path where to save the CSV file
        """
        if self.events is None:
            raise ValueError("Events must be compiled first. Call compile_major_events() method.")
        
        self.events.to_csv(filepath, index=False)
        print(f"Events saved to {filepath}")
    
    def get_events_by_category(self, category):
        """
        Get events filtered by category.
        
        Parameters:
        -----------
        category : str
            Event category to filter by
            
        Returns:
        --------
        pd.DataFrame
            Filtered events DataFrame
        """
        if self.events is None:
            raise ValueError("Events must be compiled first. Call compile_major_events() method.")
        
        return self.events[self.events['category'] == category]
    
    def get_events_by_period(self, start_date, end_date):
        """
        Get events within a specific date range.
        
        Parameters:
        -----------
        start_date : str or datetime
            Start date for filtering
        end_date : str or datetime
            End date for filtering
            
        Returns:
        --------
        pd.DataFrame
            Filtered events DataFrame
        """
        if self.events is None:
            raise ValueError("Events must be compiled first. Call compile_major_events() method.")
        
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        
        mask = (self.events['date'] >= start_date) & (self.events['date'] <= end_date)
        return self.events[mask]
    
    def print_events_summary(self):
        """
        Print a summary of the compiled events.
        """
        if self.events is None:
            raise ValueError("Events must be compiled first. Call compile_major_events() method.")
        
        print("=" * 80)
        print("GEOPOLITICAL EVENTS AFFECTING OIL PRICES (1987-2022)")
        print("=" * 80)
        
        # Summary by category
        print("\nEvents by Category:")
        category_counts = self.events['category'].value_counts()
        for category, count in category_counts.items():
            print(f"  {category}: {count} events")
        
        # Summary by expected impact
        print("\nEvents by Expected Impact:")
        impact_counts = self.events['expected_impact'].value_counts()
        for impact, count in impact_counts.items():
            print(f"  {impact}: {count} events")
        
        # List all events
        print("\nDetailed Event List:")
        print("-" * 80)
        for idx, row in self.events.iterrows():
            print(f"{idx+1:2d}. {row['date'].strftime('%Y-%m-%d')} - {row['event']}")
            print(f"     Category: {row['category']}")
            print(f"     Expected Impact: {row['expected_impact']}")
            print(f"     Description: {row['description']}")
            print()
    
    def get_events_for_analysis(self, oil_data_start_date, oil_data_end_date):
        """
        Get events that fall within the oil price data period.
        
        Parameters:
        -----------
        oil_data_start_date : datetime
            Start date of oil price data
        oil_data_end_date : datetime
            End date of oil price data
            
        Returns:
        --------
        pd.DataFrame
            Events within the data period
        """
        if self.events is None:
            raise ValueError("Events must be compiled first. Call compile_major_events() method.")
        
        # Filter events within the oil data period
        mask = (self.events['date'] >= oil_data_start_date) & (self.events['date'] <= oil_data_end_date)
        relevant_events = self.events[mask].copy()
        
        print(f"Found {len(relevant_events)} events within the oil price data period")
        print(f"Oil data period: {oil_data_start_date.strftime('%Y-%m-%d')} to {oil_data_end_date.strftime('%Y-%m-%d')}")
        
        return relevant_events


def main():
    """
    Main function to run the event research.
    """
    # Initialize the researcher
    researcher = OilEventResearcher()
    
    # Compile events
    events = researcher.compile_major_events()
    
    # Print summary
    researcher.print_events_summary()
    
    # Save to CSV
    researcher.save_events_to_csv()
    
    return researcher


if __name__ == "__main__":
    main() 