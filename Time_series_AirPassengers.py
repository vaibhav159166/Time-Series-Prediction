# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 15:33:29 2024
@author: Vaibhav Bhorkade
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('dark_background')

# Load the dataset
df=pd.read_csv("AirPassengers.csv")
df.columns

df=df.rename({'#Passengers':'Passengers'},axis=1)

print(df.dtypes)
# Month is text and passengers in int
# Now let us convert into date and time
df['Month']=pd.to_datetime(df['Month'])
df.dtypes

df.set_index('Month',inplace=True)

plt.plot(df.Passengers)
# There is increasing trend and it has got seasonality

# Is the data stationary
# Dickey-Fuller test
from statsmodels.tsa.stattools import adfuller
adf,pvalue,usedlag_,nobs_,critical_values_,icbest_=adfuller(df)
print('pvalue= ',pvalue,"if above 0.05 , data is not stationary")
# Since data is not stationary, we may need SARIMA and not just ARIMA
# Now let us extract the year and month from the date time column

df['year']=[d.year for d in df.index]
df['month']=[d.strftime("%b") for d in df.index]
years=df['year'].unique()

# Plot yearly and monthly values as boxplot
sns.boxplot(x='year',y='Passengers',data=df)
# No. of passengers are going up year by year
sns.boxplot(x='month',y='Passengers',data=df)
# Over all there is higher trend in July and August compared to rest of the

# Extract and plot trend , seasonal and residuals
from statsmodels.tsa.seasonal import seasonal_decompose
decomposed=seasonal_decompose(df['Passengers'],model='additive')

# Additive time series
# Value = Base Level + Trend + Seasonality + Error
# Multiplicative Time series
# Value = Base Level X Trend X Seasonality X Error

trend=decomposed.trend
seasonal=decomposed.seasonal # Cyclic behavior may not be seasonal
residual=decomposed.resid

plt.figure(figsize=(12,8))
plt.subplot(411)
plt.plot(['Passengers'],label="Original",color='yellow')
plt.legend(loc='upper left')
plt.subplot(412)
plt.plot(trend,label="Original",color='yellow')
plt.legend(loc='upper left')
plt.subplot(413)
plt.plot(seasonal,label="Original",color='yellow')
plt.legend(loc='upper left')
plt.plot(414)
plt.plot(residual,label="Original",color='yellow')
plt.legend(loc='upper left')
plt.show()

'''
Trend is going up from 1950 s to 60 s 
It is highly seasonal showing peaks at particular interval
This helps to select specific predication model
'''

# Autocorrelation
# values are not correlated with x-axis but with its lag
# meaning yesturdays value depend on day before yesturday so on forth
# Autocorrelation is simply the correlation of a series with its own lags
# Plot lag on x axis and correlations on y axis
# Any correlations above confidence lnes are statistically significant.

from statsmodels.tsa.stattools import acf

acf_144=acf(df.Passengers,nlags=144)
plt.plot(acf_144)

# Auto correlations above zero means positive correlations and below as negative 
# Obtain the same but with single line and more info...

from pandas.plotting import autocorrelation_plot
autocorrelation_plot(df.Passengers)

# Any lag before 40 has positive correlations
# horizontal bands indicate 95% and 99% (dashed) confidence bands
