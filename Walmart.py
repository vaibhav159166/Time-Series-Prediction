# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 16:41:07 2024
@author: Vaibhav
"""

import pandas as pd
import statsmodels.graphics.tsaplots as tsa_plots
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
from matplotlib import pyplot

Walmart=pd.read_csv("Walmart Footfalls Raw.csv")
# Data Partition
Train=Walmart.head(147)
Test=Walmart.tail(12)
# In order to use this model , we need to first find out values
# p represents number of Autoregressive terms - lags of dependant variable
# q represents number of moving Average terms
# lagged forecast errors in prediction equation
# d represents number of non-seasonal differences
# To find 

tsa_plots.plot_acf(Walmart.Footfalls, lags=12) # q for MA 5

tsa_plots.plot_pacf(Walmart.Footfalls, lags=12) # p for AR

# ARIMA with AR=3, MA=5
model1=ARIMA(Train.Footfalls,order=(3,1,5))
res1=model1.fit()
print(res1.summary())

# Forecast for next 12 months
start_index=len(Train)
end_index=start_index+11
forecast_test=res1.predict(start=start_index,end=end_index)

print(forecast_test)

# Evaluate forecasts
rmse_test=sqrt(mean_squared_error(Test.Footfalls,forecast_test))
print('Test RMSE: %.3f'% rmse_test)

# plot forecasts against actual outcomes
pyplot.plot(Test.Footfalls)
pyplot.plot(forecast_test,color='red')
pyplot.show()

# Auto-ARIMA - Automatically discover the optimal order for an ARIMA
# pip install pmdarima -- user

import pmdarima as pm

ar_model=pm.auto_arima(Train.Footfalls,start_p=0,start_q=0,max_p=12,max_q=12,m=1,d=None,seasonal=False,start_P=0,trace=True,error_action='warn',stepwise=True)