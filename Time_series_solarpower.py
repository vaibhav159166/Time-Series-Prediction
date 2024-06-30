# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 19:42:49 2024
@author: Vaibhav Bhorkade
"""

"""
Solar power consumption has been recorded by city councils at regular intervals.
The reason behind doing so is to understand how businesses are using solar power
so that they can cut down on nonrenewable sources of energy and shift towards 
renewable energy. Based on the data, build a forecasting model and provide 
insights on it. 
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # for visualization
from statsmodels.formula.api import smf

# Read CSV data
df = pd.read_csv("solarpower_cumuldaybyday2.csv")

# Explore data (head, tail, shape, describe, data types)
print(df.head())
print(df.tail())
print(df.shape)
print(df.describe())
print(df.dtypes)

# Extract month (first 3 characters from 'Month' column)
df['month'] = df['Month'].str[:3]

# Create dummy variables for months
month_dummies = pd.get_dummies(df['month'], prefix='M')  # Add prefix for clarity

# Create time trend features
df['t'] = np.arange(1, len(df) + 1)
df['t_squared'] = df['t'] ** 2

# Logarithmic transformation (optional, consider based on data exploration)
# df['log_Sales'] = np.log(df['Sales'])

# Split data into training and testing sets (80% training, 20% testing)
train_size = int(len(df) * 0.8)
train, test = df.iloc[:train_size], df.iloc[train_size:]

# Define models (consider adding ARIMA or Prophet for exploration)
models = [
    ("Linear", "Sales ~ t", train),
    ("Quadratic", "Sales ~ t + t_squared", train),
    ("Additive Seasonal", "Sales ~ " + " + ".join(month_dummies.columns), train.merge(month_dummies, left_index=True, right_index=True)),
    ("Multiplicative Seasonal (Log)", "log_Sales ~ " + " + ".join(month_dummies.columns), train.merge(month_dummies, left_index=True, right_index=True)),
    ("Additive Seasonal + Quadratic", "Sales ~ t + t_squared ~ " + " + ".join(month_dummies.columns), train.merge(month_dummies, left_index=True, right_index=True)),
]

# Fit models, calculate, and compare RMSE
rmse_results = []
for name, formula, data in models:
    model = smf.ols(formula, data=data).fit()
    predictions = model.predict(test.merge(month_dummies, left_index=True, right_index=True) if 'M_' in formula else test)
    rmse = np.sqrt(np.mean((test['Sales'] - predictions) ** 2))
    rmse_results.append((name, rmse))

# Print RMSE results
print("\nRMSE Results:")
for name, rmse in rmse_results:
    print(f"{name}: {rmse:.4f}")

# Select best model based on lowest RMSE
best_model_name, best_rmse = min(rmse_results, key=lambda x: x[1])
print(f"\nBest Model: {best_model_name} (RMSE: {best_rmse:.4f})")

# Train the best model on entire dataset
full_model = smf.ols(models[rmse_results.index(min(rmse_results, key=lambda x: x[1]))][1], df.merge(month_dummies, left_index=True, right_index=True)).fit()

# Forecast for next year (adjust horizon as needed)
#future_data = pd.DataFrame({'t': np.arange(len(df) + 1, len(df) + 13), 'M_' + m: 0 for m in month_dummies.columns})
#future_predictions = full_model.predict(future_data)

# Print or plot forecasts (consider plotting actual vs. predicted)
print("\nForecasts for next year:")
#print(future_predictions)

# Optional: Visualize actual vs. predicted values
plt.figure(figsize=(10, 6))
plt.plot(df.index, df['Sales'], label='Actual Sales')
#plt.plot(test.index, future_predictions[: len(test)], label='Predicted Sales')  # Adjust based on test data size
plt.xlabel('Month')
plt.ylabel('Sales')
plt.title('df Sales Forecast')
plt.legend()
plt.grid(True)
plt.show()