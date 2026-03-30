Python 3.13.12 (tags/v3.13.12:1cbe481, Feb  3 2026, 18:22:25) [MSC v.1944 64 bit (AMD64)] on win32
Enter "help" below or click "Help" above for more information.

======= RESTART: C:\Users\dell\OneDrive\Desktop\New folder\aiml--true.py =======
Loading dataset...
Shape       :  (52416, 9)
Columns     :  ['Datetime', 'Temperature', 'Humidity', 'WindSpeed', 'GeneralDiffuseFlows', 'DiffuseFlows', 'PowerConsumption_Zone1', 'PowerConsumption_Zone2', 'PowerConsumption_Zone3']

First 3 rows:

        Datetime  Temperature  ...  PowerConsumption_Zone2  PowerConsumption_Zone3
0  1/1/2017 0:00        6.559  ...             16128.87538             20240.96386
1  1/1/2017 0:10        6.414  ...             19375.07599             20131.08434
2  1/1/2017 0:20        6.313  ...             19006.68693             19668.43373

[3 rows x 9 columns]

Missing values:

Datetime                  0
Temperature               0
Humidity                  0
WindSpeed                 0
GeneralDiffuseFlows       0
DiffuseFlows              0
PowerConsumption_Zone1    0
PowerConsumption_Zone2    0
PowerConsumption_Zone3    0
dtype: int64

Date range  : 
2017-01-01 00:00:00
-> 
2017-12-30 23:50:00
Total rows  : 52416

Rows after feature engineering: 52272

Train samples : 41817
Test  samples :  10455

Training Linear Regression
Training Random Forest (may take ~30 seconds)


MODEL EVALUATION

========================================
  Linear Regression
========================================

  x  (Mean Absolute Error)        : 534.6760748194685  kWh
  y (Root Mean Squared Error)    :  822.167348656173  kWh
  z (Mean Abs. Percentage Error) :  0.8321420962574309 %

========================================
  Random Forest
========================================

  x  (Mean Absolute Error)        : 468.37559012203786  kWh
  y (Root Mean Squared Error)    :  726.1804525660085  kWh
  z (Mean Abs. Percentage Error) :  0.7329339867758655 %


Feature Importance:
lag_1                  0.9915
lag_144                0.0036
lag_6                  0.0020
GeneralDiffuseFlows    0.0009
hour                   0.0005
DiffuseFlows           0.0005
rolling_avg_1h         0.0004
Temperature            0.0001
Humidity               0.0001
day                    0.0001
dtype: float64
