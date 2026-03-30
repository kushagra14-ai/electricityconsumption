import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split


print("Loading dataset...")
df = pd.read_csv("powerconsumption.csv")

print("Shape       : ",df.shape)
print("Columns     : ",df.columns.tolist())
print()
print("First 3 rows:")
print()
print(df.head(3))
print()
print("Missing values:")
print()
print(df.isnull().sum())

df["Datetime"] = pd.to_datetime(df["Datetime"])
df = df.sort_values("Datetime").reset_index(drop=True)

# Combine all 3 zones into one total consumption column
df["TotalConsumption"] = (
    df["PowerConsumption_Zone1"] +
    df["PowerConsumption_Zone2"] +
    df["PowerConsumption_Zone3"]
)

print()
print("Date range  : ")
print(df['Datetime'].min())
print("-> ")
print(df['Datetime'].max())
print(f"Total rows  : {len(df)}")

df["hour"]        = df["Datetime"].dt.hour
df["minute"]      = df["Datetime"].dt.minute
df["day"]         = df["Datetime"].dt.day
df["month"]       = df["Datetime"].dt.month
df["dayofweek"]   = df["Datetime"].dt.dayofweek   # 0=Monday, 6=Sunday
df["is_weekend"]  = (df["dayofweek"] >= 5).astype(int)

# Lag features (using 10-min intervals: 1 lag = 10 min ago, 144 lags = 1 day ago)
df["lag_1"]       = df["TotalConsumption"].shift(1)    # 10 min ago
df["lag_6"]       = df["TotalConsumption"].shift(6)    # 1 hour ago
df["lag_144"]     = df["TotalConsumption"].shift(144)  # 24 hours ago

# Rolling average over last 1 hour (6 intervals of 10 min)
df["rolling_avg_1h"] = df["TotalConsumption"].rolling(window=6).mean()

# Drop rows with NaN from lag/rolling
df = df.dropna().reset_index(drop=True)
print()
print("Rows after feature engineering:" ,len(df))

features = [
    "Temperature", "Humidity", "WindSpeed",
    "GeneralDiffuseFlows", "DiffuseFlows",
    "hour", "minute", "day", "month",
    "dayofweek", "is_weekend",
    "lag_1", "lag_6", "lag_144", "rolling_avg_1h"
]

X = df[features]
y = df["TotalConsumption"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)
print()
print("Train samples :" ,len(X_train))
print("Test  samples : ",len(X_test))

print()
print("Training Linear Regression")
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_preds = lr.predict(X_test)

print("Training Random Forest (may take ~30 seconds)")
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)


def evaluate(name, actual, predicted):
    x  = mean_absolute_error(actual, predicted)
    y = np.sqrt(mean_squared_error(actual, predicted))
    z = np.mean(np.abs((actual - predicted) / actual)) * 100
    print()
    print('='*40)
    print(f"  {name}")
    print('='*40)
    print()
    print("  x  (Mean Absolute Error)        :" ,(x)," kWh")
    print("  y (Root Mean Squared Error)    : " ,(y)," kWh")
    print("  z (Mean Abs. Percentage Error) : ",(z),"%")

print()
print()
print("MODEL EVALUATION")
evaluate("Linear Regression", y_test, lr_preds)
evaluate("Random Forest",     y_test, rf_preds)


importance = pd.Series(rf.feature_importances_, index=features)
importance = importance.sort_values(ascending=False)

print()
print()
print("Feature Importance:")
print(importance.head(10).round(4))

n = 300
test_dates = df["Datetime"].iloc[len(y_train) : len(y_train) + n]

fig, axes = plt.subplots(3, 1, figsize=(15, 13))
fig.suptitle("Electricity Consumption Forecasting -- powerconsumption.csv",
             fontsize=14, fontweight="bold")

# --- Plot 1: Linear Regression ---
axes[0].plot(test_dates.values, y_test.values[:n],
             label="Actual", color="blue", linewidth=1.5)
axes[0].plot(test_dates.values, lr_preds[:n],
             label="Linear Regression", color="red",
             linewidth=1.2, linestyle="--")
axes[0].set_title("Linear Regression (Actual vs Predicted)")
axes[0].set_ylabel("Total Consumption (kWh)")
axes[0].legend(); axes[0].grid(alpha=0.3)
axes[0].tick_params(axis="x", rotation=30)

# --- Plot 2: Random Forest ---
axes[1].plot(test_dates.values, y_test.values[:n],
             label="Actual", color="blue", linewidth=1.5)
axes[1].plot(test_dates.values, rf_preds[:n],
             label="Random Forest", color="green",
             linewidth=1.2, linestyle="--")
axes[1].set_title("Random Forest (Actual vs Predicted)")
axes[1].set_ylabel("Total Consumption (kWh)")
axes[1].legend(); axes[1].grid(alpha=0.3)
axes[1].tick_params(axis="x", rotation=30)

# --- Plot 3: Feature Importance ---
axes[2].bar(importance.index, importance.values,
            color="purple", edgecolor="white")
axes[2].set_title("Feature Importance (Random Forest)")
axes[2].set_ylabel("Importance Score")
axes[2].set_xlabel("Feature")
axes[2].tick_params(axis="x", rotation=45)
axes[2].grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("forecast_results.png", dpi=150, bbox_inches="tight")
plt.show()
print()
print("Plot saved as forecast_results.png")

print()
print("===== CUSTOM PREDICTION EXAMPLE =====")

sample_input = pd.DataFrame([{
    "Temperature"         : 15.0,
    "Humidity"            : 60.0,
    "WindSpeed"           : 0.5,
    "GeneralDiffuseFlows" : 0.1,
    "DiffuseFlows"        : 0.05,
    "hour"                : 18,       # 6 PM
    "minute"              : 0,
    "day"                 : 10,
    "month"               : 6,        # June
    "dayofweek"           : 0,        # Monday
    "is_weekend"          : 0,
    "lag_1"               : 60000,    # 10 min ago
    "lag_6"               : 58000,    # 1 hour ago
    "lag_144"             : 62000,    # yesterday same time
    "rolling_avg_1h"      : 59000     # last 1 hour average
}])

rf_pred = rf.predict(sample_input)[0]
lr_pred = lr.predict(sample_input)[0]

print("  Scenario       : Monday 6 PM, June, Temp=15 degC, Humidity=60%")
print("  LR  Prediction :", (lr_pred), "kWh")
print("  RF  Prediction :",(rf_pred)," kWh")
print()
print("Done")
