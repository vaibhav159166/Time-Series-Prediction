# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 16:47:25 2024
@author: Vaibhav Bhorkade
"""
"""
A plastics manufacturing plant has recorded their monthly sales data from 1949 
to 1953. Perform forecasting on the data and bring out insights from it and 
forecast the sale for the next year. 
Plastic Sales.csv
"""
import pandas as pd
import numpy as np

month=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
plastic=pd.read_csv("PlasticSales.csv")

plastic['Month'][0]
p=plastic['Month'][0]
p
p[0:3]

plastic['month']=0

for i in range(60):
    p=plastic['Month'][i]
    plastic['month'][i]=p[0:3]
    
dummy= pd.DataFrame(pd.get_dummies(plastic['month']))

plastic1=pd.concat((plastic,dummy),axis=1)


t=np.arange(1,61)
plastic1['t']=t
t_square=plastic1['t']*plastic1['t']
plastic1['t_square']=t_square

log_Sales=np.log(plastic1['Sales'])

plastic1['log_Sales']=log_Sales

train=plastic1.head(48)
test=plastic1.tail(12)

plastic1.Sales.plot()

import statsmodels.formula.api as smf
#linear model
linear_model= smf.ols('Sales~t',data=train).fit()
linear_model
predlinear= pd.Series(linear_model.predict(pd.DataFrame(test['t'])))
rmse_lin= np.sqrt(np.mean((np.array(test['Sales'])-np.array(predlinear))**2))
rmse_lin

#quadratic model
quad_model=smf.ols('Sales~t+t_square',data=train).fit()
predquad=pd.Series(quad_model.predict(pd.DataFrame(test[['t','t_square']])))
predquad

rmse_quad=np.sqrt(np.mean((np.array(test['Sales'])-np.array(predquad))**2))

#exponential model
exp_model=smf.ols('log_Sales~t',data=train).fit()
predexp=pd.Series(exp_model.predict(pd.DataFrame(test['t'])))
rmse_exp= np.sqrt(np.mean((np.array(test['Sales'])-np.array(np.exp(predexp)))**2))
rmse_exp

#additive seasonality
add_sea=smf.ols('Sales~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov+Dec',data=train).fit()
pred_addsea=pd.Series(add_sea.predict(pd.DataFrame(test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']])))
rmse_add= np.sqrt(np.mean((np.array(test['Sales'])-np.array(pred_addsea))**2))
rmse_add
#additve with linear
add_sealin=smf.ols('Sales~t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov+Dec',data=train).fit()
predaddlin=pd.Series(add_sealin.predict(pd.DataFrame(test[['t','Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']])))
rmseaddlin= np.sqrt(np.mean((np.array(test['Sales'])-np.array(predaddlin))**2))
rmseaddlin

#additive with quadratic
add_seaquad = smf.ols('Sales~t+t_square+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov+Dec',data=train).fit()
predaddquad= pd.Series(add_seaquad.predict(test[['t','t_square','Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']]))
rmseaddquad= np.sqrt(np.mean((np.array(test['Sales'])-np.array(predaddquad))**2))
rmseaddquad

#multiplicative seasonaity
mul_lin= smf.ols('log_Sales~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov+Dec',data=train).fit()
predmul=pd.Series(mul_lin.predict(test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']]))
rmsemul=np.sqrt(np.mean((np.array(test['Sales'])-np.array(np.exp(predmul)))**2))
rmsemul

#multiplicative additive seasonality
mul_add= smf.ols('log_Sales~t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov+Dec',data=train).fit()
predmuladd= pd.Series(mul_add.predict(test[['t','Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']]))
rmsemuladd = np.sqrt(np.mean((np.array(test['Sales'])-np.array(np.exp(predmuladd)))**2))
rmsemuladd

#multiplicative additive quadratic
mul_quad = smf.ols('log_Sales~t+t_square+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov+Dec',data=train).fit()
predmulquad= pd.Series(mul_add.predict(test[['t','t_square','Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']]))
rmsemulquad = np.sqrt(np.mean((np.array(test['Sales'])-np.array(np.exp(predmulquad)))**2))
rmsemulquad

#tabular form of rmse
data={'MODEL': pd.Series(['rmse_add','rmse_exp','rmse_lin','rmse_quad','rmseaddlin','rmseaddquad','rmsemul','rmsemuladd','rmsemulquad']), 'ERROR_VALUES':pd.Series([rmse_add,rmse_exp,rmse_lin,rmse_quad,rmseaddlin,rmseaddquad,rmsemul,rmsemuladd,rmsemulquad])}
table_rmse= pd.DataFrame(data)
table_rmse
#final model is 

finalmodel =smf.ols('Sales~t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov+Dec',data=plastic1).fit()

