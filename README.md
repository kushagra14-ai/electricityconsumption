**Smart Grid Electricity Consumption Forecasting**


This project leverages machine learning to predict energy demand by analyzing historical consumption patterns and environmental factors. By utilizing both linear and non-linear regression techniques, the system identifies key drivers of power usage across three distinct zones.


 Table of Contents
•	Data Architecture
•	Feature Engineering Pipeline
•	Model Methodology
•	Evaluation Metrics
•	Visual Analytics
•	Usage & Deployment
________________________________________
** Data Architecture**
The project utilizes the powerconsumption.csv dataset, which records energy usage at 10-minute intervals.


1. Data Integration
Instead of analyzing zones in isolation, the script calculates a Total Consumption metric by aggregating three separate zones:
$TotalConsumption = Zone_1 + Zone_2 + Zone_3$

3. Temporal Features
The raw Datetime string is parsed into individual components to help the models understand seasonality and human behavior:
•	Daily Cycles: Hour and Minute.
•	Weekly Cycles: Day of week and a binary is_weekend flag (detecting Saturday/Sunday).
•	Yearly Cycles: Month and Day.
________________________________________

 
** Feature Engineering Pipeline**
The script employs advanced time-series techniques to provide the models with historical context (short-term and long-term memory).
Lag Features (The "Memory" of the Model)
•	lag_1: Consumption from 10 minutes ago.
•	lag_6: Consumption from 1 hour ago.
•	lag_144: Consumption from exactly 24 hours ago (capturing daily periodicity).
Rolling Statistics
•	rolling_avg_1h: A 6-interval windowed average that smooths out high-frequency noise and captures the immediate trend of the last hour.

________________________________________
**
 Model Methodology**
 
Two distinct approaches are compared to evaluate the trade-off between simplicity and predictive power:
Linear Regression (Baseline)
•	Type: Parametric model.
•	Purpose: Establishes a baseline by assuming a linear relationship between features (like temperature) and power demand.
Random Forest Regressor (Advanced)
•	Type: Ensemble of 100 Decision Trees.
•	Strengths: Captures complex, non-linear interactions (e.g., the way high humidity impacts power differently on weekends vs. weekdays).
•	Configuration: Uses n_jobs=-1 for parallel processing to speed up training.
________________________________________

Visual Analytics
The script automatically generates a comprehensive dashboard (forecast_results.png):
1.	Prediction vs. Actual (LR): Shows how well the linear model follows the peaks and valleys.
2.	Prediction vs. Actual (RF): Highlights the Random Forest's ability to fit tight curves and sudden spikes.
3.	Feature Importance: A bar chart ranking the top 10 most influential variables.
o	Note: Typically, historical lags (lag_144) and hour of the day are the highest contributors.
________________________________________


**Usage & Deployment**
Custom Prediction Example
The script includes a "Scenario Simulator" where you can input specific conditions (e.g., a cold Monday evening in June) to see what the models predict:

**output**
<img width="428" height="378" alt="image" src="https://github.com/user-attachments/assets/44c24016-9ade-4f3a-8f87-51ffb7b25dd2" />


# Scenario:
LR Prediction : 59231.42 kWh
RF Prediction : 60124.85 kWh

 

Setup Requirements
1.	Place powerconsumption.csv in the root directory.
2.	Install dependencies: pip install numpy pandas matplotlib scikit-learn
3.	Execute: python aiml--true.py

