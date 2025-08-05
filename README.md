# Brent Oil Price Change Point Analysis

## Project Overview

This project analyzes how geopolitical events affect Brent oil prices using Bayesian change point detection and statistical modeling. The analysis focuses on identifying structural breaks in oil price time series and correlating them with major political and economic events.

## Business Objective

The main goal is to study how important events affect Brent oil prices, focusing on:
- Political decisions
- Conflicts in oil-producing regions  
- Global economic sanctions
- OPEC policy changes

This analysis provides insights for investors, analysts, and policymakers to better understand and react to price changes.

## Project Structure

```
├── data/                          # Raw and processed data
│   └── BrentOilPrices.csv        # Historical Brent oil prices (1987-2022)
├── src/                           # Source code
│   ├── analysis/                  # Data analysis scripts
│   ├── models/                    # Bayesian change point models
│   └── dashboard/                 # Web dashboard
│       ├── backend/               # Flask API
│       └── frontend/              # React frontend
├── notebooks/                     # Jupyter notebooks for exploration
├── reports/                       # Analysis reports and documentation
├── docs/                          # Project documentation
└── instructions/                  # Project requirements and materials
```

## Key Features

### Task 1: Foundation Analysis
- Data exploration and time series analysis
- Event research and compilation
- Workflow definition and assumptions

### Task 2: Change Point Detection
- Bayesian change point modeling with PyMC3
- MCMC sampling for posterior inference
- Event correlation and impact quantification

### Task 3: Interactive Dashboard
- Flask backend API
- React frontend with interactive visualizations
- Event highlighting and filtering capabilities

## Technical Stack

- **Python**: PyMC3, pandas, numpy, matplotlib, arviz
- **Bayesian Methods**: MCMC sampling, change point detection
- **Web**: Flask (backend), React (frontend)
- **Visualization**: Interactive charts and dashboards

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd b5w10-Change-Point-Analysis-and-Statistical-Modelling-of-Time-Series-Data

# Install Python dependencies
pip install -r requirements.txt

# Install Node.js dependencies (for dashboard)
cd src/dashboard/frontend
npm install
```

## Usage

### Data Analysis
```bash
# Run data exploration
python src/analysis/explore_data.py

# Run change point detection
python src/models/change_point_detection.py
```

### Dashboard
```bash
# Start Flask backend
cd src/dashboard/backend
python app.py

# Start React frontend
cd src/dashboard/frontend
npm start
```

## Key Dates

- **Interim Submission**: Sunday, Aug 1, 2025 (20:00 UTC)
- **Final Submission**: Tuesday, Aug 5, 2025 (20:00 UTC)

## Team

- **Tutors**: Mahlet, Rediet, Kerod, Rehmet
- **Student**: Tinbite Yonas

## References

- [Data Science Workflow](https://www.datascience-pm.com/data-science-workflow/)
- [Change Point Detection](https://forecastegy.com/posts/change-point-detection-time-series-python/)
- [Bayesian Changepoint Detection with PyMC3](https://www.pymc.io/blog/chris_F_pydata2022.html)
- [React Dashboard Templates](https://github.com/flatlogic/react-dashboard) 