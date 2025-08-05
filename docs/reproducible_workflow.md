# Reproducible Data Analysis Workflow

## Brent Oil Price Change Point Analysis Project

### Executive Summary

This document defines a comprehensive, reproducible workflow for analyzing how geopolitical events affect Brent oil prices using Bayesian change point detection. The workflow ensures consistency, transparency, and reproducibility across all analysis stages.

---

## 1. Workflow Overview

### 1.1 Workflow Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Input    │───▶│  Processing &   │───▶│   Analysis &    │
│   & Validation  │    │   Exploration   │    │   Modeling      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Event Data    │    │   Statistical   │    │   Change Point  │
│   Compilation   │    │   Analysis      │    │   Detection     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Correlation   │    │   Visualization │    │   Results &     │
│   Analysis      │    │   & Reporting   │    │   Documentation │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### 1.2 Workflow Stages

1. **Data Foundation** (Task 1)
2. **Change Point Modeling** (Task 2)
3. **Dashboard Development** (Task 3)

---

## 2. Stage 1: Data Foundation Workflow

### 2.1 Data Loading and Validation

**Step 1.1: Data Source Setup**
```python
# File: src/analysis/data_exploration.py
def load_and_validate_data():
    """
    Load Brent oil price data and perform initial validation
    """
    # Load data from CSV
    df = pd.read_csv('data/BrentOilPrices.csv')
    
    # Validate data structure
    assert 'Date' in df.columns, "Date column missing"
    assert 'Price' in df.columns, "Price column missing"
    assert len(df) > 0, "Dataset is empty"
    
    return df
```

**Step 1.2: Data Quality Checks**
```python
def perform_data_quality_checks(df):
    """
    Comprehensive data quality assessment
    """
    # Check for missing values
    missing_data = df.isnull().sum()
    
    # Check for duplicates
    duplicates = df.duplicated().sum()
    
    # Check date range
    date_range = df['Date'].min(), df['Date'].max()
    
    # Check price validity
    price_stats = df['Price'].describe()
    
    return {
        'missing_data': missing_data,
        'duplicates': duplicates,
        'date_range': date_range,
        'price_stats': price_stats
    }
```

### 2.2 Data Preprocessing Pipeline

**Step 1.3: Data Cleaning and Transformation**
```python
def preprocess_oil_data(df):
    """
    Standardized data preprocessing pipeline
    """
    # Convert dates
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%b-%y')
    
    # Sort by date
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Set date as index
    df.set_index('Date', inplace=True)
    
    # Calculate log returns
    df['log_returns'] = np.log(df['Price'] / df['Price'].shift(1))
    
    # Remove NaN values
    df_clean = df.dropna()
    
    return df_clean
```

### 2.3 Event Data Compilation

**Step 1.4: Event Research and Compilation**
```python
# File: src/analysis/event_research.py
def compile_geopolitical_events():
    """
    Compile comprehensive event dataset
    """
    researcher = OilEventResearcher()
    events_df = researcher.get_events_dataframe()
    
    # Save to CSV
    events_df.to_csv('data/geopolitical_events.csv', index=False)
    
    return events_df
```

### 2.4 Statistical Analysis Pipeline

**Step 1.5: Exploratory Data Analysis**
```python
def comprehensive_eda(df):
    """
    Complete exploratory data analysis
    """
    # Basic statistics
    basic_stats = df.describe()
    
    # Stationarity tests
    stationarity_results = perform_stationarity_tests(df)
    
    # Volatility analysis
    volatility_analysis = analyze_volatility_patterns(df)
    
    # Distribution analysis
    distribution_analysis = analyze_distributions(df)
    
    return {
        'basic_stats': basic_stats,
        'stationarity': stationarity_results,
        'volatility': volatility_analysis,
        'distributions': distribution_analysis
    }
```

---

## 3. Stage 2: Change Point Modeling Workflow

### 3.1 Model Preparation

**Step 2.1: Data Preparation for Modeling**
```python
# File: src/models/change_point_detection.py
def prepare_data_for_modeling(df, use_log_returns=True):
    """
    Prepare data for Bayesian change point detection
    """
    if use_log_returns:
        data = df['log_returns'].values
        dates = df.index
    else:
        data = df['Price'].values
        dates = df.index
    
    return data, dates
```

### 3.2 Bayesian Model Specification

**Step 2.2: Single Change Point Model**
```python
def build_single_change_point_model(data):
    """
    Build Bayesian single change point model using PyMC3
    """
    with pm.Model() as model:
        # Priors
        mu1 = pm.Normal('mu1', mu=0, sd=1)
        mu2 = pm.Normal('mu2', mu=0, sd=1)
        sigma = pm.HalfNormal('sigma', sd=1)
        
        # Change point
        tau = pm.DiscreteUniform('tau', lower=0, upper=len(data)-1)
        
        # Likelihood
        mu = pm.math.switch(tau >= np.arange(len(data)), mu1, mu2)
        likelihood = pm.Normal('likelihood', mu=mu, sd=sigma, observed=data)
    
    return model
```

**Step 2.3: Multiple Change Point Model**
```python
def build_multiple_change_point_model(data, max_changepoints=5):
    """
    Build Bayesian multiple change point model
    """
    with pm.Model() as model:
        # Number of change points
        n_changepoints = pm.DiscreteUniform('n_changepoints', 
                                          lower=0, upper=max_changepoints)
        
        # Parameters for each segment
        mus = pm.Normal('mus', mu=0, sd=1, shape=max_changepoints+1)
        sigma = pm.HalfNormal('sigma', sd=1)
        
        # Change point locations
        taus = pm.DiscreteUniform('taus', lower=0, upper=len(data)-1, 
                                shape=max_changepoints)
        
        # Likelihood
        likelihood = pm.Normal('likelihood', mu=mus[0], sd=sigma, observed=data)
    
    return model
```

### 3.3 MCMC Sampling and Inference

**Step 2.4: Model Fitting and Sampling**
```python
def fit_change_point_model(model, data, draws=2000, tune=1000):
    """
    Fit change point model using MCMC sampling
    """
    with model:
        # Run MCMC
        trace = pm.sample(draws=draws, tune=tune, return_inferencedata=True)
        
        # Posterior predictive sampling
        ppc = pm.sample_posterior_predictive(trace, samples=1000)
    
    return trace, ppc
```

### 3.4 Results Analysis and Interpretation

**Step 2.5: Change Point Detection and Analysis**
```python
def analyze_change_point_results(trace, dates, events_df):
    """
    Analyze MCMC results and correlate with events
    """
    # Extract change point estimates
    changepoint_dates = extract_changepoint_dates(trace, dates)
    
    # Calculate parameter estimates
    parameter_estimates = extract_parameter_estimates(trace)
    
    # Correlate with events
    event_correlations = correlate_with_events(changepoint_dates, events_df)
    
    return {
        'changepoint_dates': changepoint_dates,
        'parameter_estimates': parameter_estimates,
        'event_correlations': event_correlations
    }
```

---

## 4. Stage 3: Visualization and Reporting Workflow

### 4.1 Automated Visualization Pipeline

**Step 3.1: Time Series Visualization**
```python
def create_time_series_plots(df, changepoints, events_df):
    """
    Create comprehensive time series visualizations
    """
    # Main price plot with change points
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.plot(df.index, df['Price'], linewidth=1, alpha=0.8)
    
    # Add change points
    for cp in changepoints:
        ax.axvline(x=cp, color='red', linestyle='--', alpha=0.7)
    
    # Add event markers
    for _, event in events_df.iterrows():
        ax.axvline(x=event['date'], color='orange', linestyle=':', alpha=0.5)
    
    ax.set_title('Brent Oil Prices with Detected Change Points and Events')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (USD per barrel)')
    
    return fig
```

**Step 3.2: Statistical Diagnostic Plots**
```python
def create_diagnostic_plots(trace, model):
    """
    Create MCMC diagnostic plots
    """
    # Trace plots
    pm.plot_trace(trace)
    
    # Posterior distributions
    pm.plot_posterior(trace)
    
    # Autocorrelation plots
    pm.autocorrplot(trace)
    
    # Summary statistics
    summary = pm.summary(trace)
    
    return summary
```

### 4.2 Results Compilation and Reporting

**Step 3.3: Automated Report Generation**
```python
def generate_analysis_report(results, config):
    """
    Generate comprehensive analysis report
    """
    report = {
        'executive_summary': create_executive_summary(results),
        'methodology': document_methodology(config),
        'results': compile_results(results),
        'visualizations': create_visualizations(results),
        'conclusions': generate_conclusions(results),
        'recommendations': provide_recommendations(results)
    }
    
    return report
```

---

## 5. Reproducibility Framework

### 5.1 Environment Management

**Environment Setup**
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import pymc3, pandas, numpy; print('All packages installed successfully')"
```

### 5.2 Configuration Management

**Configuration File: `config/analysis_config.yaml`**
```yaml
# Data configuration
data:
  input_file: "data/BrentOilPrices.csv"
  events_file: "data/geopolitical_events.csv"
  output_dir: "results/"

# Model configuration
model:
  use_log_returns: true
  max_changepoints: 5
  mcmc_draws: 2000
  mcmc_tune: 1000

# Analysis configuration
analysis:
  confidence_level: 0.95
  event_correlation_window: 30  # days
  volatility_window: 30  # days

# Output configuration
output:
  save_plots: true
  save_results: true
  report_format: "html"
```

### 5.3 Version Control and Documentation

**Git Workflow**
```bash
# Initialize repository
git init
git add .
git commit -m "Initial project setup"

# Create feature branches
git checkout -b task-1
git checkout -b task-2
git checkout -b task-3

# Regular commits with clear messages
git commit -m "Add data exploration pipeline"
git commit -m "Implement Bayesian change point detection"
git commit -m "Create visualization dashboard"
```

### 5.4 Automated Testing

**Test Suite: `tests/test_workflow.py`**
```python
import unittest
import pandas as pd
import numpy as np

class TestWorkflow(unittest.TestCase):
    
    def test_data_loading(self):
        """Test data loading and validation"""
        df = load_and_validate_data()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)
    
    def test_preprocessing(self):
        """Test data preprocessing pipeline"""
        df = load_and_validate_data()
        df_processed = preprocess_oil_data(df)
        self.assertFalse(df_processed.isnull().any().any())
    
    def test_event_compilation(self):
        """Test event data compilation"""
        events_df = compile_geopolitical_events()
        self.assertIsInstance(events_df, pd.DataFrame)
        self.assertGreater(len(events_df), 0)

if __name__ == '__main__':
    unittest.main()
```

---

## 6. Quality Assurance and Validation

### 6.1 Data Quality Checks

**Automated Quality Checks**
```python
def run_quality_checks():
    """
    Run comprehensive quality checks
    """
    checks = {
        'data_completeness': check_data_completeness(),
        'data_consistency': check_data_consistency(),
        'model_convergence': check_model_convergence(),
        'result_validation': validate_results()
    }
    
    return all(checks.values()), checks
```

### 6.2 Model Validation

**Cross-Validation Framework**
```python
def validate_change_point_model(data, dates, events_df):
    """
    Validate change point model performance
    """
    # Split data for validation
    train_data, test_data = split_data(data, dates)
    
    # Fit model on training data
    model = build_single_change_point_model(train_data)
    trace = fit_change_point_model(model, train_data)
    
    # Validate on test data
    validation_results = validate_on_test_data(trace, test_data)
    
    return validation_results
```

---

## 7. Execution Workflow

### 7.1 Main Execution Script

**File: `src/main_analysis.py`**
```python
def main():
    """
    Main execution workflow
    """
    print("Starting Brent Oil Price Change Point Analysis...")
    
    # Stage 1: Data Foundation
    print("\n=== Stage 1: Data Foundation ===")
    df = load_and_validate_data()
    df_processed = preprocess_oil_data(df)
    events_df = compile_geopolitical_events()
    
    # Stage 2: Change Point Modeling
    print("\n=== Stage 2: Change Point Modeling ===")
    data, dates = prepare_data_for_modeling(df_processed)
    model = build_single_change_point_model(data)
    trace, ppc = fit_change_point_model(model, data)
    
    # Stage 3: Analysis and Reporting
    print("\n=== Stage 3: Analysis and Reporting ===")
    results = analyze_change_point_results(trace, dates, events_df)
    report = generate_analysis_report(results, config)
    
    print("Analysis complete!")
    return results, report

if __name__ == "__main__":
    results, report = main()
```

### 7.2 Command Line Interface

**File: `run_analysis.py`**
```python
#!/usr/bin/env python3
"""
Command line interface for running the analysis
"""
import argparse
import sys
from src.main_analysis import main

def parse_arguments():
    parser = argparse.ArgumentParser(description='Brent Oil Price Change Point Analysis')
    parser.add_argument('--stage', choices=['1', '2', '3', 'all'], 
                       default='all', help='Analysis stage to run')
    parser.add_argument('--config', default='config/analysis_config.yaml',
                       help='Configuration file path')
    parser.add_argument('--output', default='results/',
                       help='Output directory')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    try:
        results, report = main()
        print("Analysis completed successfully!")
    except Exception as e:
        print(f"Error during analysis: {e}")
        sys.exit(1)
```

---

## 8. Documentation and Communication

### 8.1 Automated Documentation

**Documentation Generation**
```python
def generate_documentation():
    """
    Generate comprehensive documentation
    """
    docs = {
        'methodology': generate_methodology_doc(),
        'results': generate_results_doc(),
        'technical': generate_technical_doc(),
        'user_guide': generate_user_guide()
    }
    
    return docs
```

### 8.2 Stakeholder Communication

**Communication Templates**
```python
def create_stakeholder_report(results, audience='executive'):
    """
    Create stakeholder-specific reports
    """
    if audience == 'executive':
        return create_executive_summary(results)
    elif audience == 'technical':
        return create_technical_report(results)
    elif audience == 'public':
        return create_public_report(results)
```

---

## 9. Maintenance and Updates

### 9.1 Regular Updates

**Update Schedule**
- **Weekly**: Data quality checks and validation
- **Monthly**: Model performance review and updates
- **Quarterly**: Full workflow review and optimization
- **Annually**: Complete methodology review and documentation update

### 9.2 Version Control

**Version Management**
```python
VERSION = "1.0.0"
LAST_UPDATED = "2024-01-01"
CHANGELOG = {
    "1.0.0": "Initial release with basic change point detection",
    "1.1.0": "Added multiple change point detection",
    "1.2.0": "Enhanced visualization and reporting"
}
```

---

## 10. Conclusion

This reproducible workflow ensures that our analysis of Brent oil price change points is:

1. **Consistent**: Standardized procedures across all analysis stages
2. **Transparent**: Clear documentation of all methods and assumptions
3. **Reproducible**: Complete environment and data management
4. **Scalable**: Modular design allows for easy expansion and modification
5. **Quality-Assured**: Comprehensive testing and validation procedures

The workflow provides a solid foundation for data-driven insights into the relationship between geopolitical events and oil price movements, while maintaining the highest standards of scientific rigor and reproducibility.

---

*Workflow Version: 1.0*  
*Last Updated: [Current Date]*  
*Next Review: [Date + 3 months]* 