# Assumptions and Limitations Document

## Brent Oil Price Change Point Analysis Project

### Executive Summary

This document outlines the key assumptions, limitations, and methodological considerations for our analysis of how geopolitical events affect Brent oil prices using Bayesian change point detection. Understanding these factors is crucial for interpreting results and communicating findings to stakeholders.

---

## 1. Core Assumptions

### 1.1 Data Quality Assumptions

**Assumption 1: Data Completeness**
- The Brent oil price dataset (1987-2022) is assumed to be complete and accurate
- Missing values, if any, are assumed to be random and not systematic
- The data source is reliable and represents actual market transactions

**Assumption 2: Market Efficiency**
- Oil markets are assumed to be reasonably efficient in incorporating new information
- Price movements reflect the collective assessment of market participants
- There are no significant market manipulation or data reporting biases

**Assumption 3: Event Data Reliability**
- Geopolitical events are accurately dated and categorized
- Event impact assessments are based on historical analysis and expert consensus
- Event descriptions capture the primary drivers of price movements

### 1.2 Methodological Assumptions

**Assumption 4: Change Point Model Validity**
- The Bayesian change point model adequately captures structural breaks in the time series
- The underlying statistical assumptions (e.g., normality of log returns) are reasonable
- The model can distinguish between noise and genuine structural changes

**Assumption 5: Temporal Relationship**
- Changes in oil prices following events occur within a reasonable timeframe
- The relationship between events and price changes is not instantaneous but occurs within days to weeks
- There is a causal relationship between major geopolitical events and oil price movements

**Assumption 6: Independence of Events**
- Major geopolitical events are assumed to be independent of each other
- The impact of one event does not systematically influence the impact of subsequent events
- Event clustering does not significantly bias the analysis

---

## 2. Critical Limitations

### 2.1 Correlation vs. Causation

**Limitation 1: Correlation Does Not Imply Causation**
- **Critical Limitation**: Our analysis identifies correlations between events and price changes, not causal relationships
- **Implication**: We cannot definitively state that specific events caused specific price movements
- **Mitigation**: We use multiple analytical approaches and acknowledge this limitation in all communications

**Limitation 2: Confounding Variables**
- **Issue**: Many factors influence oil prices simultaneously (economic conditions, weather, technological changes)
- **Impact**: Isolating the effect of any single event is challenging
- **Mitigation**: We categorize events by type and analyze patterns across similar events

### 2.2 Data Limitations

**Limitation 3: Event Selection Bias**
- **Issue**: We focus on major, well-documented events, potentially missing smaller but significant events
- **Impact**: May underestimate the cumulative effect of smaller events
- **Mitigation**: We acknowledge this limitation and focus on high-impact events

**Limitation 4: Price Data Granularity**
- **Issue**: Daily price data may miss intraday volatility and immediate market reactions
- **Impact**: Some short-term effects may be smoothed out
- **Mitigation**: We use log returns and rolling volatility measures to capture short-term dynamics

**Limitation 5: Event Impact Measurement**
- **Issue**: Quantifying the exact impact of events is subjective and based on historical analysis
- **Impact**: Impact assessments may vary among analysts
- **Mitigation**: We use standardized impact levels and provide detailed descriptions

### 2.3 Model Limitations

**Limitation 6: Model Specification**
- **Issue**: The Bayesian change point model makes specific assumptions about the data generating process
- **Impact**: Model results depend on these assumptions being approximately correct
- **Mitigation**: We test model assumptions and use multiple model specifications

**Limitation 7: Parameter Uncertainty**
- **Issue**: Model parameters have uncertainty that propagates to change point estimates
- **Impact**: Change point dates have confidence intervals, not exact dates
- **Mitigation**: We report uncertainty measures and confidence intervals

**Limitation 8: Stationarity Assumptions**
- **Issue**: The model assumes log returns are stationary
- **Impact**: Long-term trends may affect change point detection
- **Mitigation**: We test for stationarity and use appropriate transformations

---

## 3. Methodological Considerations

### 3.1 Statistical Considerations

**Consideration 1: Multiple Testing**
- **Issue**: Testing multiple events increases the chance of false positives
- **Mitigation**: We use appropriate significance levels and multiple comparison corrections

**Consideration 2: Model Selection**
- **Issue**: Choosing the number of change points involves trade-offs between model fit and complexity
- **Mitigation**: We use information criteria and cross-validation approaches

**Consideration 3: Outlier Handling**
- **Issue**: Extreme price movements may be outliers or genuine structural changes
- **Mitigation**: We use robust statistical methods and carefully examine extreme observations

### 3.2 Temporal Considerations

**Consideration 4: Event Timing**
- **Issue**: The exact timing of events and their market impact may not align perfectly
- **Mitigation**: We use flexible time windows and examine lead-lag relationships

**Consideration 5: Market Adaptation**
- **Issue**: Markets may adapt to certain types of events over time
- **Mitigation**: We analyze whether event impacts have changed over different time periods

**Consideration 6: Seasonality**
- **Issue**: Oil prices exhibit seasonal patterns that may confound event analysis
- **Mitigation**: We account for seasonality in our models where appropriate

---

## 4. Communication Guidelines

### 4.1 Stakeholder Communication

**Guideline 1: Clear Language**
- Always use "correlation" rather than "causation" when describing relationships
- Emphasize that results are probabilistic, not deterministic
- Use confidence intervals and uncertainty measures in communications

**Guideline 2: Context Provision**
- Provide historical context for events and their market significance
- Explain the limitations of the analysis clearly
- Acknowledge alternative explanations for observed patterns

**Guideline 3: Appropriate Cautions**
- Include disclaimers about the limitations of the analysis
- Emphasize that results should not be used as the sole basis for investment decisions
- Recommend consultation with financial advisors for investment decisions

### 4.2 Reporting Standards

**Standard 1: Transparency**
- Report all model assumptions and limitations
- Provide detailed methodology descriptions
- Include confidence intervals and uncertainty measures

**Standard 2: Reproducibility**
- Document all data sources and processing steps
- Provide code and methodology for independent verification
- Maintain clear version control for all analyses

**Standard 3: Regular Updates**
- Update assumptions and limitations as new information becomes available
- Revise impact assessments based on new research
- Maintain currency of event database

---

## 5. Risk Management

### 5.1 Analysis Risks

**Risk 1: Overinterpretation**
- **Risk**: Drawing overly strong conclusions from correlational analysis
- **Mitigation**: Maintain strict adherence to correlation vs. causation distinction

**Risk 2: Model Misspecification**
- **Risk**: Using inappropriate models for the data structure
- **Mitigation**: Test multiple model specifications and validate assumptions

**Risk 3: Data Quality Issues**
- **Risk**: Undetected errors in price or event data
- **Mitigation**: Implement data quality checks and validation procedures

### 5.2 Communication Risks

**Risk 4: Misleading Stakeholders**
- **Risk**: Stakeholders may misinterpret results as causal relationships
- **Mitigation**: Clear communication of limitations and appropriate disclaimers

**Risk 5: Overconfidence**
- **Risk**: Presenting results with unwarranted certainty
- **Mitigation**: Emphasize uncertainty and probabilistic nature of results

---

## 6. Recommendations for Use

### 6.1 Appropriate Uses

✅ **Suitable Applications:**
- Understanding historical patterns in oil price movements
- Identifying periods of significant market volatility
- Informing risk management strategies
- Supporting scenario analysis and stress testing
- Providing context for market analysis

❌ **Inappropriate Uses:**
- Making precise predictions about future oil prices
- Determining exact causal relationships between events and prices
- Making investment decisions without additional analysis
- Ignoring other market fundamentals and factors

### 6.2 Best Practices

**Practice 1: Holistic Analysis**
- Combine change point analysis with fundamental analysis
- Consider multiple data sources and analytical approaches
- Integrate qualitative and quantitative insights

**Practice 2: Regular Review**
- Update analyses as new data becomes available
- Reassess assumptions and limitations periodically
- Incorporate new research and market developments

**Practice 3: Stakeholder Education**
- Educate stakeholders about the limitations of the analysis
- Provide training on appropriate interpretation of results
- Maintain open communication about methodology and assumptions

---

## 7. Conclusion

This analysis provides valuable insights into the relationship between geopolitical events and Brent oil price movements. However, it is essential to understand and communicate the limitations and assumptions underlying the analysis. The results should be used as one component of a comprehensive market analysis framework, not as standalone predictive tools.

**Key Takeaway**: While we can identify patterns and correlations between events and price movements, we must be cautious about inferring causal relationships. The analysis serves as a tool for understanding market dynamics and informing risk management strategies, but should be used in conjunction with other analytical approaches and expert judgment.

---

*Document Version: 1.0*  
*Last Updated: [Current Date]*  
*Next Review: [Date + 6 months]* 