# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 08:36:32 2024
@author: Vaibhav Bhorkade
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import Holt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Now to load the datasets
cocacola=pd.read_excel("CocaCola_Sales_Rawdata.xlsx")
# Let us plot the dataset and its nature
cocacola.Sales.plot()
# Spliting the data into train and Test set data
# since we are working on quarterly datasets and in year there 
# Test data=4 quarters
# train data = 38
Train=cocacola.head(38)
Test=cocacola.tail(4)
# Here we are considering performance parameters as mean absolute
# rather than mean square error
# custom function is written to calculate MPSE
def MAPE(pred,org):
    temp=np.abs((pred-org)/org)*100
    return np.mean(temp)
# EDA which comprices identification of level , trends and seasonal
# In order to seprate Trend and Seasonality moving average 
mv_pred=cocacola['Sales'].rolling(4).mean()
mv_pred.tail(4)

# now let us calcualate mean absolute percentage of these values
# basic purpose pf moving average is deseasonalizing
cocacola.Sales.plot(label="org")
# This is original plot
# Now let us seperate out trend and seasonality
for i in range(2,9,2):
    # it will take window size 2,4,6,8
    cocacola["Sales"].rolling(i).mean().plot(label=str(i))
    plt.legend(loc=3)
# you can see i=4 and 8 are deseasonable plots

# Time series decomposition is the another technique of seperating
# seasonality
decompose_ts_add=seasonal_decompose(cocacola.Sales,model="additive",period=4)
print(decompose_ts_add.trend)
print(decompose_ts_add.seasonal)
print(decompose_ts_add.resid)
print(decompose_ts_add.observed)

decompose_ts_add.plot()

# Similar plot can be decomposed using multiplicative
decompose_ts_mul=seasonal_decompose(cocacola.Sales,model="multiplicative",period=4)

print(decompose_ts_mul.trend)
print(decompose_ts_mul.seasonal)
print(decompose_ts_mul.resid)
print(decompose_ts_mul.observed)
decompose_ts_mul.plot()
# you can observe the difference between these plots
# Now let us plot ACF plot to check the auto correlation
import statsmodels.graphics.tsaplots as tsa_plots
tsa_plots.plot_acf(cocacola.Sales,lags=4)
# we can observe the output in which r1,r2,r3 and r4 has higher 
### This is all about EDA
# Let us apply data to data driven models
# Simple exponential method
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
ses_model=SimpleExpSmoothing(Train["Sales"]).fit()
pred_ses=ses_model.predict(start=Test.index[0],end=Test.index[-1])

# now calculate MAPE
MAPE(pred_ses,Test.Sales)
# we are getting 8.3698
# Holts exponential smoothing # here only trend is captured
hw_model=Holt(Train["Sales"]).fit()
pred_hw=hw_model.predict(start=Test.index[0],end=Test.index[-1])
MAPE(pred_hw,Test.Sales)
# 10.485

# Holts winter exponential smoothing with additive seasonality
hwe_model_add_add=ExponentialSmoothing(Train["Sales"],seasonal="add",trend="add",seasonal_periods=4).fit()
pred_hwe_model_add_add=hwe_model_add_add.predict(start=Test.index[0],end=Test.index[-1])

MAPE(pred_hwe_model_add_add,Test.Sales)
# 1.5023
# Holts winter exponential smoothing with multiplicative seasonal

hwe_model_mul_add=ExponentialSmoothing(Train["Sales"],seasonal='mul',trend="add",seasonal_periods=4).fit()
pred_hwe_model_mul_add=hue_model_mul_add.predict(strat=Test.index[0],end=Test.index[-1])

MAPE(pred_hwe_model_mil_add,Test.Sales)









