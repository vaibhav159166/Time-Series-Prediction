# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 15:11:33 2024
@author: Vaibhav Bhorkade
"""

import pandas as pd
Walmart=pd.read_csv("Walmart Footfalls Raw.csv")

# pre-processing
import numpy as np

Walmart["t"] = np.arange(1,160)

Walmart["t_squared"] = Walmart["t"]*Walmart["t"]
Walmart["log_footfalls"] = np.log(Walmart["Footfalls"])
Walmart.columns

#month =['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
#In Walmart data we have Jan-1991 in the 0th column, we need only first three 
#example- Jan from each cell

p = Walmart["Month"][0]
#Before we will extract , let us create new column called
#month to store extracted values
p[0:3]

Walmart['months']= 0
#You can check the dataframe with months name with all values=0
#the total records are 159 in walmart

for i in range(159):
    p = Walmart["Month"][i]
    Walmart['months'][i]= p[0:3]
    
month_dummies = pd.DataFrame(pd.get_dummies(Walmart['months']))
# Now let us concatnate these dummy values to dataframe
Walmart1 = pd.concat([Walmart, month_dummies], axis = 1)
#you can check the dataframe walmart1

# Visualization
Walmart1.Footfalls.plot()

# Data Partition
Train = Walmart1.head(147)
Test = Walmart1.tail(12)

# to change the index value in pandas data frame 
# Test.set_index(np.arange(1,13))

####################### Linear ##########################
import statsmodels.formula.api as smf 

linear_model = smf.ols('Footfalls ~ t', data=Train).fit()
pred_linear =  pd.Series(linear_model.predict(pd.DataFrame(Test['t'])))
rmse_linear = np.sqrt(np.mean((np.array(Test['Footfalls']) - np.array(pred_linear))**2))
rmse_linear

##################### Exponential ##############################

Exp = smf.ols('log_footfalls ~ t', data = Train).fit()
pred_Exp = pd.Series(Exp.predict(pd.DataFrame(Test['t'])))
rmse_Exp = np.sqrt(np.mean((np.array(Test['Footfalls']) - np.array(np.exp(pred_Exp)))**2))
rmse_Exp

#################### Quadratic ###############################
 
Quad = smf.ols('Footfalls ~ t+t_squared', data=Train).fit()
pred_Quad = pd.Series(Quad.predict(Test[["t","t_squared"]]))
rmse_Quad = np.sqrt(np.mean((np.array(Test['Footfalls'])-np.array(pred_Quad))**2))
rmse_Quad

################### Additive seasonality ########################

add_sea = smf.ols('Footfalls ~ Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov', data=Train).fit()
pred_add_sea = pd.Series(add_sea.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov']]))
rmse_add_sea = np.sqrt(np.mean((np.array(Test['Footfalls'])-np.array(pred_add_sea))**2))
rmse_add_sea

################## Multiplicative Seasonality ##################

Mul_sea = smf.ols('log_footfalls ~ Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data = Train).fit()
pred_Mult_sea = pd.Series(Mul_sea.predict(Test))
rmse_Mult_sea = np.sqrt(np.mean((np.array(Test['Footfalls']) - np.array(np.exp(pred_Mult_sea)))**2))
rmse_Mult_sea

################## Additive Seasonality Quadratic Trend ############################

add_sea_Quad = smf.ols('Footfalls ~ t+t_squared+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Train).fit()
pred_add_sea_quad = pd.Series(add_sea_Quad.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','t','t_squared']]))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(Test['Footfalls'])-np.array(pred_add_sea_quad))**2))
rmse_add_sea_quad 

################## Multiplicative Seasonality Linear Trend  ###########

Mul_Add_sea = smf.ols('log_footfalls ~ t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data = Train).fit()
pred_Mult_add_sea = pd.Series(Mul_Add_sea.predict(Test))
rmse_Mult_add_sea = np.sqrt(np.mean((np.array(Test['Footfalls'])-np.array(np.exp(pred_Mult_add_sea)))**2))
rmse_Mult_add_sea 
################# Additive Seasonality Quadratic Trend ################

add_sea_Quad=smf.ols('Footfalls ~ Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov+t+t_squared',data=Train).fit()
pred_add_sea_Quad=pd.Series(add_sea_Quad.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','t','t_squared']]))
rmse_add_sea_Quad=np.sqrt(np.mean((np.array(Test['Footfalls'])-np.array(pred_add_sea_Quad))**2))

rmse_add_sea_Quad

################## Multiplicative Seasonality ###################

Mul_sea=smf.ols('log_footfalls ~ Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Train).fit()
pred_Mult_sea=pd.Series(Mul_sea.predict(Test))
rmse_Mult_sea=np.sqrt(np.mean((np.array(Test['Footfalls'])-np.array(np.exp(pred_Mult_sea)))**2))

rmse_Mult_sea


################## Testing / consolidate #######################################

data = {"MODEL":pd.Series(["rmse_linear","rmse_Exp","rmse_Quad","rmse_add_sea","rmse_add_sea_quad","rmse_Mult_sea","rmse_Mult_add_sea"]),"RMSE_Values":pd.Series([rmse_linear,rmse_Exp,rmse_Quad,rmse_add_sea,rmse_add_sea_quad,rmse_Mult_sea,rmse_Mult_add_sea])}
table_rmse=pd.DataFrame(data)
table_rmse

# 'rmse_add_sea' has the least value among the models prepared so far Predicting new values 
predict_data = pd.read_excel("Predict_new.xlsx")

model_full = smf.ols('Footfalls ~ t+t_squared+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov', data=Walmart1).fit()


# Assuming 't' is already defined in the predict_data DataFrame
predict_data['t_squared'] = predict_data['t'] ** 2

# Make predictions using the model_full model
pred_new = model_full.predict(predict_data)

# Add the predictions to the predict_data DataFrame
predict_data['forecasted_Footfalls'] = pred_new

pred_new  = pd.Series(model_full.predict(predict_data))
pred_new

predict_data["forecasted_Footfalls"] = pd.Series(pred_new)

#  Autoregression model (AR)
# Calculating Residuals from best model applied on full data
# AV-FV
full_res=Walmart.Footfalls - model_full.predict(Walmart)

# ACF plot on residuals
import statsmodels.graphics.tsaplots as tsa_plots
tsa_plots.plot_acf(full_res,lags=12)

# ACF is an (complete) auto-correlation function gives values
# of auto-correlation of any  time series with its lagged values.

# PACF is a partial auto-correlation function
# It finds correlations of present with lags of the residuals of the 
tsa_plots.plot_pacf(full_res, lags=12)

# Alternative approach for ACF plot
# From pandas.plotting import autocorrelation_plot 
# autocorrelation_ppyplot.show()

# AR model
from statsmodels.tsa.ar_model import AutoReg
model_ar=AutoReg(full_res, lags=[1])
# model_ar=AutoReg(Train_res,lags=12)
model_fit=model_ar.fit()

print('Coeficient : %s'%model_fit.params)

pred_res=model_fit.predict(start=len(full_res),end=len(full_res)+len(predict_data)-1,dynamic=False)
pred_res.reset_index(drop=True,inplace=True)

# The final Predictions using ASQT and AR model
final_pred=pred_new+pred_res
final_pred


