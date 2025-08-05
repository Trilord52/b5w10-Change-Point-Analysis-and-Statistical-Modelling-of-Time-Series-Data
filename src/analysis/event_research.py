"""
Event Research Module for Brent Oil Price Analysis

This module compiles and structures major geopolitical events that have affected
Brent oil prices from 1987 to 2022. It provides a comprehensive dataset of
events with their dates, categories, descriptions, and estimated impacts.
"""

import pandas as pd
from datetime import datetime
import numpy as np

class OilEventResearcher:
    """
    A class to compile and manage geopolitical events affecting oil prices.
    
    This class maintains a structured dataset of major events that have
    influenced Brent oil prices, including their categorization and
    potential impact assessments.
    """
    
    def __init__(self):
        """Initialize the event researcher with a comprehensive event database."""
        self.events = self._compile_events()
        
    def _compile_events(self):
        """
        Compile a comprehensive list of major geopolitical events affecting oil prices.
        
        Returns:
            list: List of dictionaries containing event information
        """
        events = [
            # 1. Gulf War (1990-1991)
            {
                'date': '1990-08-02',
                'event_name': 'Iraq Invasion of Kuwait',
                'category': 'War/Conflict',
                'description': 'Iraq invades Kuwait, leading to the Gulf War and significant oil supply disruptions',
                'impact_level': 'High',
                'impact_description': 'Oil prices surged from ~$20 to ~$40 per barrel due to supply concerns',
                'affected_regions': ['Middle East', 'Global'],
                'duration_days': 210
            },
            
            # 2. Asian Financial Crisis (1997-1998)
            {
                'date': '1997-07-02',
                'event_name': 'Asian Financial Crisis Begins',
                'category': 'Economic Crisis',
                'description': 'Financial crisis in Asia leads to economic downturn and reduced oil demand',
                'impact_level': 'Medium',
                'impact_description': 'Oil prices fell from ~$25 to ~$12 per barrel due to reduced demand',
                'affected_regions': ['Asia', 'Global'],
                'duration_days': 365
            },
            
            # 3. 9/11 Attacks (2001)
            {
                'date': '2001-09-11',
                'event_name': '9/11 Terrorist Attacks',
                'category': 'Terrorism',
                'description': 'Terrorist attacks on the US lead to geopolitical instability and oil market uncertainty',
                'impact_level': 'High',
                'impact_description': 'Oil prices initially spiked then stabilized, marking the beginning of increased geopolitical risk premium',
                'affected_regions': ['United States', 'Global'],
                'duration_days': 30
            },
            
            # 4. Iraq War (2003)
            {
                'date': '2003-03-20',
                'event_name': 'US Invasion of Iraq',
                'category': 'War/Conflict',
                'description': 'US-led invasion of Iraq creates uncertainty about Middle East oil supplies',
                'impact_level': 'High',
                'impact_description': 'Oil prices increased from ~$30 to ~$40 per barrel due to supply concerns',
                'affected_regions': ['Middle East', 'Global'],
                'duration_days': 2555
            },
            
            # 5. Hurricane Katrina (2005)
            {
                'date': '2005-08-29',
                'event_name': 'Hurricane Katrina',
                'category': 'Natural Disaster',
                'description': 'Hurricane Katrina devastates Gulf Coast, disrupting US oil production and refining',
                'impact_level': 'Medium',
                'impact_description': 'Oil prices spiked from ~$60 to ~$70 per barrel due to supply disruptions',
                'affected_regions': ['United States', 'Gulf of Mexico'],
                'duration_days': 30
            },
            
            # 6. Global Financial Crisis (2008)
            {
                'date': '2008-09-15',
                'event_name': 'Lehman Brothers Bankruptcy',
                'category': 'Economic Crisis',
                'description': 'Global financial crisis leads to economic recession and dramatic fall in oil demand',
                'impact_level': 'Very High',
                'impact_description': 'Oil prices collapsed from ~$140 to ~$40 per barrel due to economic downturn',
                'affected_regions': ['Global'],
                'duration_days': 730
            },
            
            # 7. Arab Spring (2011)
            {
                'date': '2011-01-25',
                'event_name': 'Arab Spring Begins',
                'category': 'Political Unrest',
                'description': 'Political unrest in Middle East and North Africa creates oil supply uncertainty',
                'impact_level': 'High',
                'impact_description': 'Oil prices increased from ~$90 to ~$120 per barrel due to supply concerns',
                'affected_regions': ['Middle East', 'North Africa'],
                'duration_days': 365
            },
            
            # 8. Libyan Civil War (2011)
            {
                'date': '2011-02-17',
                'event_name': 'Libyan Civil War',
                'category': 'War/Conflict',
                'description': 'Civil war in Libya disrupts oil production and exports',
                'impact_level': 'Medium',
                'impact_description': 'Oil prices increased by ~$10-15 per barrel due to Libyan supply disruption',
                'affected_regions': ['Libya', 'Mediterranean'],
                'duration_days': 240
            },
            
            # 9. US Shale Revolution (2012-2014)
            {
                'date': '2012-01-01',
                'event_name': 'US Shale Oil Boom',
                'category': 'Technology/Supply',
                'description': 'Advances in fracking technology lead to surge in US oil production',
                'impact_level': 'High',
                'impact_description': 'Oil prices remained stable despite global demand growth due to increased supply',
                'affected_regions': ['United States', 'Global'],
                'duration_days': 730
            },
            
            # 10. OPEC Price War (2014-2016)
            {
                'date': '2014-11-27',
                'event_name': 'OPEC Decision Not to Cut Production',
                'category': 'OPEC Policy',
                'description': 'OPEC decides to maintain production levels despite falling prices, leading to price war',
                'impact_level': 'Very High',
                'impact_description': 'Oil prices collapsed from ~$100 to ~$30 per barrel',
                'affected_regions': ['Global'],
                'duration_days': 730
            },
            
            # 11. Saudi Arabia-Russia Oil Price War (2020)
            {
                'date': '2020-03-08',
                'event_name': 'Saudi-Russia Oil Price War',
                'category': 'OPEC Policy',
                'description': 'Saudi Arabia and Russia fail to agree on production cuts, leading to price war',
                'impact_level': 'Very High',
                'impact_description': 'Oil prices fell from ~$50 to ~$20 per barrel',
                'affected_regions': ['Global'],
                'duration_days': 30
            },
            
            # 12. COVID-19 Pandemic (2020)
            {
                'date': '2020-03-11',
                'event_name': 'COVID-19 Declared Pandemic',
                'category': 'Pandemic',
                'description': 'Global pandemic leads to unprecedented demand destruction for oil',
                'impact_level': 'Very High',
                'impact_description': 'Oil prices briefly went negative, then recovered to ~$40 per barrel',
                'affected_regions': ['Global'],
                'duration_days': 730
            },
            
            # 13. Biden Administration Energy Policies (2021)
            {
                'date': '2021-01-20',
                'event_name': 'Biden Administration Energy Policies',
                'category': 'Policy Change',
                'description': 'New US administration implements policies favoring renewable energy over fossil fuels',
                'impact_level': 'Medium',
                'impact_description': 'Oil prices increased due to reduced US production and regulatory uncertainty',
                'affected_regions': ['United States', 'Global'],
                'duration_days': 365
            },
            
            # 14. Russia-Ukraine Conflict (2022)
            {
                'date': '2022-02-24',
                'event_name': 'Russia Invades Ukraine',
                'category': 'War/Conflict',
                'description': 'Russian invasion of Ukraine leads to sanctions and energy supply disruptions',
                'impact_level': 'Very High',
                'impact_description': 'Oil prices surged from ~$90 to ~$120 per barrel due to supply concerns',
                'affected_regions': ['Europe', 'Global'],
                'duration_days': 220
            },
            
            # 15. OPEC+ Production Cuts (2022)
            {
                'date': '2022-10-05',
                'event_name': 'OPEC+ Announces Major Production Cuts',
                'category': 'OPEC Policy',
                'description': 'OPEC+ announces 2 million barrel per day production cut to support prices',
                'impact_level': 'High',
                'impact_description': 'Oil prices increased by ~$10-15 per barrel due to supply reduction',
                'affected_regions': ['Global'],
                'duration_days': 90
            }
        ]
        
        return events
    
    def get_events_dataframe(self):
        """
        Convert events list to a pandas DataFrame.
        
        Returns:
            pd.DataFrame: DataFrame containing all events with proper date formatting
        """
        df = pd.DataFrame(self.events)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        return df
    
    def save_events_to_csv(self, filepath='data/geopolitical_events.csv'):
        """
        Save events to a CSV file.
        
        Args:
            filepath (str): Path where to save the CSV file
        """
        df = self.get_events_dataframe()
        df.to_csv(filepath, index=False)
        print(f"Events saved to {filepath}")
        return df
    
    def get_events_by_category(self, category):
        """
        Filter events by category.
        
        Args:
            category (str): Category to filter by
            
        Returns:
            pd.DataFrame: Filtered events DataFrame
        """
        df = self.get_events_dataframe()
        return df[df['category'] == category]
    
    def get_events_by_impact_level(self, impact_level):
        """
        Filter events by impact level.
        
        Args:
            impact_level (str): Impact level to filter by ('Low', 'Medium', 'High', 'Very High')
            
        Returns:
            pd.DataFrame: Filtered events DataFrame
        """
        df = self.get_events_dataframe()
        return df[df['impact_level'] == impact_level]
    
    def get_events_in_date_range(self, start_date, end_date):
        """
        Filter events within a specific date range.
        
        Args:
            start_date (str): Start date in 'YYYY-MM-DD' format
            end_date (str): End date in 'YYYY-MM-DD' format
            
        Returns:
            pd.DataFrame: Filtered events DataFrame
        """
        df = self.get_events_dataframe()
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        return df[(df['date'] >= start_dt) & (df['date'] <= end_dt)]
    
    def get_event_summary(self):
        """
        Generate a summary of all events.
        
        Returns:
            dict: Summary statistics about the events
        """
        df = self.get_events_dataframe()
        
        summary = {
            'total_events': len(df),
            'date_range': f"{df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}",
            'categories': df['category'].value_counts().to_dict(),
            'impact_levels': df['impact_level'].value_counts().to_dict(),
            'high_impact_events': len(df[df['impact_level'].isin(['High', 'Very High'])]),
            'war_conflict_events': len(df[df['category'] == 'War/Conflict']),
            'economic_crisis_events': len(df[df['category'] == 'Economic Crisis']),
            'opec_policy_events': len(df[df['category'] == 'OPEC Policy'])
        }
        
        return summary
    
    def print_event_summary(self):
        """Print a formatted summary of all events."""
        summary = self.get_event_summary()
        
        print("="*60)
        print("GEOPOLITICAL EVENTS SUMMARY (1987-2022)")
        print("="*60)
        print(f"Total Events: {summary['total_events']}")
        print(f"Date Range: {summary['date_range']}")
        print(f"High Impact Events: {summary['high_impact_events']}")
        print(f"War/Conflict Events: {summary['war_conflict_events']}")
        print(f"Economic Crisis Events: {summary['economic_crisis_events']}")
        print(f"OPEC Policy Events: {summary['opec_policy_events']}")
        
        print("\nEvents by Category:")
        for category, count in summary['categories'].items():
            print(f"  {category}: {count}")
        
        print("\nEvents by Impact Level:")
        for level, count in summary['impact_levels'].items():
            print(f"  {level}: {count}")
        
        print("="*60)
    
    def get_events_for_analysis(self):
        """
        Get events formatted for change point analysis correlation.
        
        Returns:
            pd.DataFrame: Events DataFrame with analysis-ready format
        """
        df = self.get_events_dataframe()
        
        # Add impact score for quantitative analysis
        impact_scores = {
            'Low': 1,
            'Medium': 2,
            'High': 3,
            'Very High': 4
        }
        
        df['impact_score'] = df['impact_level'].map(impact_scores)
        
        # Add year and month for easier filtering
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        
        return df

def main():
    """Main function to demonstrate the event research functionality."""
    
    # Initialize the event researcher
    researcher = OilEventResearcher()
    
    # Print summary
    researcher.print_event_summary()
    
    # Save events to CSV
    events_df = researcher.save_events_to_csv()
    
    # Display first few events
    print("\nFirst 5 Events:")
    print(events_df.head())
    
    # Show events by category
    print("\nWar/Conflict Events:")
    war_events = researcher.get_events_by_category('War/Conflict')
    print(war_events[['date', 'event_name', 'impact_level']])
    
    # Show high impact events
    print("\nVery High Impact Events:")
    high_impact = researcher.get_events_by_impact_level('Very High')
    print(high_impact[['date', 'event_name', 'category']])
    
    return researcher

if __name__ == "__main__":
    main() 