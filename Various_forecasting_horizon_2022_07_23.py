# -*- coding: utf-8 -*-
"""
Created on Sat Jul 23 12:33:18 2022

@author: odiao
"""
import warnings
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.genmod.families import Poisson, NegativeBinomial, Gaussian
from statsmodels.genmod.families.links import identity, log, sqrt
import matplotlib.pyplot as plt
import math 

import os
#os.chdir ('C:\\Users\\odiao\\Desktop\\Model paludisme')
os.chdir('C:\\Users\\odiao\\Dropbox\\Model paludisme')

df = pd.read_csv('Dakar.csv', header=0, infer_datetime_format=False, parse_dates=[0])

#***************************************************************************************************
#Defined the avaerage temperature
AVT_Dakar=(df.Tempmin_Dakar + df.Tempmax_Dakar)/2
AVT_Fatick=(df.Tempmin_Fatick + df.Tempmax_Fatick)/2
AVT_Kedougou=(df.Tempmin_Kedougou + df.Tempmax_Kedougou)/2

warnings.filterwarnings("ignore") 
#*********************************************************************************************************************
#******************************************************************************************************
# ***********************************************************************************
#*************************************************************************************
#******************Define the forecasting function*******************************************************************
# Define function that gathers statistics:
def accuracy_measures(sw_regions,sw_model,sw_link,sw_show_results,mu_fitted, mu_forecasted, y_train, y_test, t_i, t_c, t_e, y_o,min_fitted):
    if sw_show_results=='all_links':
        stats_all=np.full((3,9), np.nan)
    if sw_show_results=='each_link':
        stats_all=np.full(9, np.nan)
    R2_train=np.corrcoef(y_train, mu_fitted)[0][1]**2
    R2_test=np.corrcoef(y_test, mu_forecasted)[0][1]**2
    RMSE_train = np.linalg.norm(mu_fitted - y_train) / math.sqrt(t_c-t_i)
    RMSE_test = np.linalg.norm(mu_forecasted - y_test) / math.sqrt(t_e-t_c)
    MAE_train = np.mean(np.abs(y_train - mu_fitted))
    MAE_test =  np.mean(np.abs(y_test - mu_forecasted))
    MASE_train = MAE_train / np.mean(np.abs(y_o[(t_i+1):t_c] - y_o[t_i:(t_c-1)]))
    MASE_test = MAE_test / np.mean(np.abs(y_o[t_c+1:t_e] - y_o[t_c:t_e-1]))  # MAE divided by MAE of the prescient naive one-step-ahead predictor (that predicts total_in[t] by total_in[t-1]). Since the decrease is slow, this can be interpreted as the MAE divided by the noise level. If it gets below 1, then the fit is visually excellent. This measure is strongly inspired from Hyndman & Koehler 2006 (https://doi.org/10.1016/j.ijforecast.2006.03.001).
    MARE_train = np.mean(np.abs((mu_fitted -  y_train) /  y_train))
    MARE_test = np.mean(np.abs((mu_forecasted - y_test) / y_test))
    Min = min_fitted
    stats_all=[RMSE_train,RMSE_test,MASE_train,MASE_test,MARE_train,MARE_test,R2_train,R2_test,Min] 
    if sw_model==Gaussian:
        sw_model='Gausian' 
    if sw_model==Poisson:
        sw_model='Poisson'
    if sw_model==NegativeBinomial:
        sw_model='NB' 
    if sw_link==identity:
        sw_link='id'
    if sw_link==log:
        sw_link='log'
    if sw_link==sqrt:
        sw_link='sqrt'
    print(sw_regions,"&",sw_model,"&",sw_link, "&", round(stats_all[0],2),"/",round(stats_all[1],2), '&', round(stats_all[2],2) ,"/", round(stats_all[3],2),'&', round(stats_all[4],2) ,"/",  round(stats_all[5],2), '&', round(stats_all[6],2) ,"/",  round(stats_all[7],2), '&', round(stats_all[8],2),'\\\\')
    
    return 
#**************Define the Plots function***************************************


#************************************************************************************************************
#*****************************************************************************************************
#*************************************************************************************************
#*****************************************************************************************************************
#********** We vary the value of the forecast horizon (h) ********************************************
#************* We only use the initial set as explanatory variables ************************************************
t_i=5; t_c=84; t_e=np.shape(df)[0]; intercept=np.ones(t_e)
for sw_regions in ['Dakar', 'Fatick', 'Kedougou']:
    for h in np.arange(1,4):
        if sw_regions == 'Dakar':  
            lag=[2,2,5,1] 
            y_o = df.MC_Dakar.values
            X = np.c_[df.Rainfall, AVT_Dakar, df.Humidity_D, df.MC_Dakar, intercept]
        if sw_regions == 'Fatick':
            lag=[3,4,3,1] 
            y_o = df.MC_Fatick.values
            X = np.c_[df.Rainfall_F, AVT_Fatick, df.Humidity_F, df.MC_Fatick, intercept]      
        if sw_regions == 'Kedougou':
            lag=[2,5,2,1] 
            y_o = df.MC_Kedougou.values
            X = np.c_[df.Rainfall_K, AVT_Kedougou, df.Humidity_K, df.MC_Kedougou, intercept] 
        y_train=y_o[t_i:t_c]; y_test=y_o[t_c:t_e]
        sw_link=identity; forecasts=np.full(t_e, np.nan)
        sw_model=Poisson
        X_train=np.c_[X[t_i-max(lag[3],h):t_c-max(lag[3],h),3], X[t_i-h:t_c-h,4]] #baseline of explanatory variables
        model = sm.GLM(y_train, X_train, family=sw_model(link=sw_link)).fit()  
        predictions = model.get_prediction(X_train)
        predictions_summary_frame = predictions.summary_frame()
        for t in np.arange(t_c,t_e):
            forecasts[t]= model.params[0]*X[t-max(lag[3],h),3] + model.params[1]
        forecasted_values=forecasts[t_c:t_e]
        print("******************** FORECAST ACCURACY MEASURES - latex table ********************")
        print("************ Horizon of: ",h)
        print('\\begin{tabular}{|c|c|c|c|c|c|c|c|}')
        print('\\hline')
        #print("RMSE_train",'&',"RMSE_test",'&',  "MASE_train",'&',"MASE_test",'&', "MARE_train",'&',"MARE_test",'&', "R2_train",'&',"R2_test",'&', "Min",'\\\\')
        print('Regions & Model & Links & RMSE\_train/RMSE_test & MASE\_train/MASE\_test & MARE\_train/MARE\_test & R2\_train/R2\_test & Min \\\\')
        print('\\hline')
        accuracy_measures(sw_regions, sw_model, sw_link, 'each_link', model.mu, forecasted_values, y_train, y_test, t_i, t_c, t_e, y_o, min(model.mu))
        print('\\end{tabular}')
        #We show the plots
        plt.figure(figsize=(16,4))
        plt.subplot(121)
        plt.plot(df.Date, y_o, color = "black" ,  marker='o', markersize=8)
        plt.plot(df.Date[t_i:t_c], model.mu, color = "blue", marker='o', markersize=8)
        plt.plot(df.Date[t_i:t_c], predictions_summary_frame.iloc[:,2], "y--", linewidth=2)
        plt.plot(df.Date[t_i:t_c], predictions_summary_frame.iloc[:,3], "g--", linewidth=2)
        plt.plot(df.Date[t_c:t_e], forecasted_values, color = "red", marker='o', markersize=8)
        plt.xlabel("Date [months]", fontsize=20)
        plt.title("A ", fontsize=20) 
        plt.ylabel("Malaria incidence", fontsize=20)
        plt.legend(["Malaria incidence count", "predicted data", "mean_ci_lower", "mean_ci_upper", "forecasted data"])
        plt.subplot(122)     
        plt.plot(df.Date[t_i-max(lag[3],h):t_e-max(lag[3],h)], model.params[0]*X[t_i-max(lag[3],h):t_e-max(lag[3],h),3], color="black", label="Malaria in the past")
        plt.axhline(y=model.params[1], color="magenta", label="Intercept")
        plt.title("B", fontsize=20) 
        plt.ylabel("β_iX_j", fontsize=20)
        plt.xlabel("Date [months]", fontsize=20)
        plt.legend()
        plt.show()                    
                  
         
                      
    