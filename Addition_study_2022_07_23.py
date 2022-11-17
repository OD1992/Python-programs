# -*- coding: utf-8 -*-
"""
Created on Sat Jul 23 12:33:18 2022

@author: odiao
"""
import warnings
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
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
def forecast_functions(sw_protocols, sw_link, t_c, t_e, lag, h, coeff, X):
    forecasts=np.full(t_e, np.nan)  # set storage
    if sw_protocols=='P1':
        for t in np.arange(t_c,t_e):
            if sw_link==identity:
                forecasts[t] = coeff[0]*X[t-max(lag[0],h),0] + coeff[1]*X[t-max(lag[1],h),1] + coeff[2]*X[t-max(lag[2],h),2] + coeff[3]*X[t-max(lag[3],h),3] + coeff[4]*X[t-h,4]
            if sw_link==log:
                forecasts[t] = np.exp(coeff[0]*X[t-max(lag[0],h),0] + coeff[1]*X[t-max(lag[1],h),1] + coeff[2]*X[t-max(lag[2],h),2] + coeff[3]*X[t-max(lag[3],h),3] + coeff[4]*X[t-h,4])
            if sw_link==sqrt:
                forecasts[t] = (coeff[0]*X[t-max(lag[0],h),0] + coeff[1]*X[t-max(lag[1],h),1] + coeff[2]*X[t-max(lag[2],h),2] + coeff[3]*X[t-max(lag[3],h),3] + coeff[4]*X[t-h,4])**2
    if sw_protocols=='P2':
        for t in np.arange(t_c,t_e):
            if sw_link==identity:                                                                                                                             
                forecasts[t]=coeff[0]*X[t,0] + coeff[1]*X[t,1] + coeff[2]*X[t,2] + coeff[3]*X[t-max(lag[3],h),3] + coeff[4]*X[t-h,4]                                                                                                                                   
            if sw_link==log:
                forecasts[t]=np.exp(coeff[0]*X[t,0] + coeff[1]*X[t,1] + coeff[2]*X[t,2] + coeff[3]*X[t-max(lag[3],h),3] + coeff[4]*X[t-h,4])
            if sw_link==sqrt:
                forecasts[t] = (coeff[0]*X[t,0] + coeff[1]*X[t,1] + coeff[2]*X[t,2] + coeff[3]*X[t-max(lag[3],h),3] + coeff[4]*X[t-h,4])**2  
    return forecasts[t_c:t_e]

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
def figures(sw_protocols, y_o, t_i, t_c, t_e, lag, h, mu_fitted, mu_train_lower, mu_train_upper, mu_forecasted,params, X):
    #We show the plots
    plt.figure(figsize=(17,7))
    plt.subplot(121)
    plt.plot(df.Date, y_o, color = "black" ,  label="Malaria incidence count", marker='o', markersize=8)
    plt.plot(df.Date[t_i:t_c], mu_fitted, color = "blue", label= "predicted data",marker='o', markersize=8)
    plt.plot(df.Date[t_i:t_c], mu_train_lower, "y--", label="data_lower", linewidth=2)
    plt.plot(df.Date[t_i:t_c], mu_train_upper, "g--", label="data_lower", linewidth=2)
    plt.plot(df.Date[t_c:t_e], mu_forecasted, color = "red", label="forecasted data", marker='o', markersize=8)
    plt.xlabel("Date [months]", fontsize=20)
    plt.title("A", fontsize=20) 
    plt.ylabel("Malaria incidence", fontsize=20)
    plt.legend(fontsize=15)
    plt.subplot(122)
    if sw_protocols=='P1':
        plt.plot(df.Date[t_i-max(lag[0],h):t_e-max(lag[0],h)], params[0]*X[t_i-max(lag[0],h):t_e-max(lag[0],h),0], color="green", label="rainfall")
        plt.plot(df.Date[t_i-max(lag[1],h):t_e-max(lag[1],h)], params[1]*X[t_i-max(lag[1],h):t_e-max(lag[1],h),1], color="red", label="avg_temperature")
        plt.plot(df.Date[t_i-max(lag[2],h):t_e-max(lag[2],h)], params[2]*X[t_i-max(lag[2],h):t_e-max(lag[2],h),2], color="cyan", label="humidity")
    if sw_protocols=='P2':
        plt.plot(df.Date[t_i:t_e], params[0]*X[t_i:t_e,0], color="green", label="rainfall")
        plt.plot(df.Date[t_i:t_e], params[1]*X[t_i:t_e,1], color="red", label="avg_temperature")
        plt.plot(df.Date[t_i:t_e], params[2]*X[t_i:t_e,2], color="cyan", label="humidity")
    plt.plot(df.Date[t_i-max(lag[3],h):t_e-max(lag[3],h)], params[3]*X[t_i-max(lag[3],h):t_e-max(lag[3],h),3], color="black", label="malaria cases in the past")
    plt.axhline(y=params[4], color="magenta", label="Intercept")                           
    plt.title("B", fontsize=20)
    plt.ylabel("β_iX_j", fontsize=20)
    plt.xlabel("Date [months]", fontsize=20)
    plt.legend(fontsize=15)
    return




#************************************************************************************************************
#*****************************************************************************************************
#************ Addition variable: we consider in first the malaria in the past and the intercept as initial set **************
#********** Then, we add one by one the rest of the variables ***********************************************
def addition_study(sw_regions, var_add, sw_model):  
    t_i=5; t_c=84; t_e=np.shape(df)[0]; intercept=np.ones(t_e); h=1
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
    if var_add==['Malaria in the past']: #baseline of explanatory variables
        X_train=np.c_[X[t_i-max(lag[3],h):t_c-max(lag[3],h),3], X[t_i-h:t_c-h,4]]
        if sw_model==Poisson:
            model = sm.GLM(y_train, X_train, family=sw_model(link=sw_link)).fit()
        if sw_model==NegativeBinomial:
            model = sm.GLM(y_train, X_train, family=Poisson(link=sw_link)).fit()
            df_train_1 = pd.DataFrame({'BB_LAMBDA':model.mu, 'MC':y_train})    
            df_train_11 = pd.DataFrame({'AUX_OLS_DEP':df_train_1.apply(lambda x: ((x['MC'] - x['BB_LAMBDA'])**2 - x['BB_LAMBDA']) / x['BB_LAMBDA'], axis=1), 'BB_LAMBDA':model.mu, 'MC':y_train})    
            ols_expr = """AUX_OLS_DEP ~ BB_LAMBDA - 1"""     
            aux_olsr_results = smf.ols(ols_expr, df_train_11).fit() #Configure and fit the OLSR model:
            model = sm.GLM(y_train, X_train,family=sw_model(alpha=aux_olsr_results.params[0], link=sw_link)).fit()  
        predictions = model.get_prediction(X_train)
        predictions_summary_frame = predictions.summary_frame()
        for t in np.arange(t_c,t_e):
            forecasts[t]= model.params[0]*X[t-max(lag[3],h),3] + model.params[1]
        forecasted_values=forecasts[t_c:t_e] 
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
        
    if var_add==['Malaria in the past', 'Rainfall']: #baseline of explanatory variables
        X_train=np.c_[X[t_i-max(lag[3],h):t_c-max(lag[3],h),3], X[t_i-max(lag[0],h):t_c-max(lag[0],h),0], X[t_i-h:t_c-h,4]]
        if sw_model==Poisson:
            model = sm.GLM(y_train, X_train, family=sw_model(link=sw_link)).fit()
        if sw_model==NegativeBinomial:
            model = sm.GLM(y_train, X_train, family=Poisson(link=sw_link)).fit()
            df_train_1 = pd.DataFrame({'BB_LAMBDA':model.mu, 'MC':y_train,'Malaria in the past':X[t_i-max(lag[3],h):t_c-max(lag[3],h),3], 'Rainfall':X[t_i-max(lag[0],h):t_c-max(lag[0],h),0],'Intercept':X[t_i-h:t_c-h,4]})    
            df_train_11 = pd.DataFrame({'AUX_OLS_DEP':df_train_1.apply(lambda x: ((x['MC'] - x['BB_LAMBDA'])**2 - x['MC']) / x['BB_LAMBDA'], axis=1), 'BB_LAMBDA':model.mu, 'MC':y_train,'Malaria in the past':X[t_i-max(lag[3],h):t_c-max(lag[3],h),3],'Rainfall':X[t_i-max(lag[0],h):t_c-max(lag[0],h),0], 'Intercept':X[t_i-h:t_c-h,4]})    
            ols_expr = """AUX_OLS_DEP ~ BB_LAMBDA - 1"""     
            aux_olsr_results = smf.ols(ols_expr, df_train_11).fit() #Configure and fit the OLSR model:
            model = sm.GLM(y_train, X_train,family=sw_model(alpha=aux_olsr_results.params[0], link=sw_link)).fit()
        predictions = model.get_prediction(X_train)
        predictions_summary_frame = predictions.summary_frame()
        for t in np.arange(t_c,t_e):
            forecasts[t]= model.params[0]*X[t-max(lag[3],h),3] + model.params[1]*X[t-max(lag[0],h),0] + model.params[2]
        forecasted_values=forecasts[t_c:t_e] 
        #We show the plots
        plt.figure(figsize=(14,4))
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
        plt.plot(df.Date[t_i-max(lag[0],h):t_e-max(lag[0],h)], model.params[1]*X[t_i-max(lag[0],h):t_e-max(lag[0],h),0], color="green", label="Rainfall")
        plt.axhline(y=model.params[2], color="magenta", label="Intercept")
        plt.title("B", fontsize=20) 
        plt.ylabel("β_iX_j", fontsize=20)
        plt.xlabel("Date [months]", fontsize=20)
        plt.legend()
        plt.show()
    if var_add==['Malaria in the past', 'Temperature']: #baseline of explanatory variables
        X_train=np.c_[X[t_i-max(lag[3],h):t_c-max(lag[3],h),3], X[t_i-max(lag[1],h):t_c-max(lag[1],h),1], X[t_i-h:t_c-h,4]]
        if sw_model==Poisson:
            model = sm.GLM(y_train, X_train, family=sw_model(link=sw_link)).fit()
        if sw_model==NegativeBinomial:
            model = sm.GLM(y_train, X_train, family=Poisson(link=sw_link)).fit()
            df_train_1 = pd.DataFrame({'BB_LAMBDA':model.mu, 'MC':y_train})    
            df_train_11 = pd.DataFrame({'AUX_OLS_DEP':df_train_1.apply(lambda x: ((x['MC'] - x['BB_LAMBDA'])**2 - x['BB_LAMBDA']) / x['BB_LAMBDA'], axis=1), 'BB_LAMBDA':model.mu, 'MC':y_train})    
            ols_expr = """AUX_OLS_DEP ~ BB_LAMBDA - 1"""     
            aux_olsr_results = smf.ols(ols_expr, df_train_11).fit() #Configure and fit the OLSR model:
            model = sm.GLM(y_train, X_train,family=sw_model(alpha=aux_olsr_results.params[0], link=sw_link)).fit()
        predictions = model.get_prediction(X_train)
        predictions_summary_frame = predictions.summary_frame()
        for t in np.arange(t_c,t_e):
            forecasts[t]= model.params[0]*X[t-max(lag[3],h),3] + model.params[1]*X[t-max(lag[1],h),1] + model.params[2]
        forecasted_values=forecasts[t_c:t_e] 
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
        plt.plot(df.Date[t_i-max(lag[1],h):t_e-max(lag[1],h)], model.params[1]*X[t_i-max(lag[1],h):t_e-max(lag[1],h),1], color="red", label="Temperature")
        plt.axhline(y=model.params[2], color="magenta", label="Intercept")
        plt.title("B", fontsize=20) 
        plt.ylabel("β_iX_j", fontsize=20)
        plt.xlabel("Date [months]", fontsize=20)
        plt.legend()
        plt.show()
    if var_add==['Malaria in the past', 'Humidity']: #baseline of explanatory variables
        X_train=np.c_[X[t_i-max(lag[3],h):t_c-max(lag[3],h),3], X[t_i-max(lag[2],h):t_c-max(lag[2],h),2], X[t_i-h:t_c-h,4]]
        if sw_model==Poisson:
            model = sm.GLM(y_train, X_train, family=sw_model(link=sw_link)).fit()
        if sw_model==NegativeBinomial:
            model = sm.GLM(y_train, X_train, family=Poisson(link=sw_link)).fit()
            df_train_1 = pd.DataFrame({'BB_LAMBDA':model.mu, 'MC':y_train})    
            df_train_11 = pd.DataFrame({'AUX_OLS_DEP':df_train_1.apply(lambda x: ((x['MC'] - x['BB_LAMBDA'])**2 - x['BB_LAMBDA']) / x['BB_LAMBDA'], axis=1), 'BB_LAMBDA':model.mu, 'MC':y_train})    
            ols_expr = """AUX_OLS_DEP ~ BB_LAMBDA - 1"""     
            aux_olsr_results = smf.ols(ols_expr, df_train_11).fit() #Configure and fit the OLSR model:
            model = sm.GLM(y_train, X_train,family=sw_model(alpha=aux_olsr_results.params[0], link=sw_link)).fit()
        predictions = model.get_prediction(X_train)
        predictions_summary_frame = predictions.summary_frame()
        for t in np.arange(t_c,t_e):
            forecasts[t]= model.params[0]*X[t-max(lag[3],h),3] + model.params[1]*X[t-max(lag[2],h),2] + model.params[2]
        forecasted_values=forecasts[t_c:t_e] 
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
        plt.plot(df.Date[t_i-max(lag[2],h):t_e-max(lag[2],h)], model.params[1]*X[t_i-max(lag[2],h):t_e-max(lag[2],h),2], color="cyan", label="Humidity")
        plt.axhline(y=model.params[2], color="magenta", label="Intercept")
        plt.title("B", fontsize=20) 
        plt.ylabel("β_iX_j", fontsize=20)
        plt.xlabel("Date [months]", fontsize=20)
        plt.legend()
        plt.show()
    if var_add==['Malaria in the past', 'Rainfall', 'Humidity']: #baseline of explanatory variables
        X_train=np.c_[X[t_i-max(lag[3],h):t_c-max(lag[3],h),3], X[t_i-max(lag[0],h):t_c-max(lag[0],h),0], X[t_i-max(lag[2],h):t_c-max(lag[2],h),2], X[t_i-h:t_c-h,4]]
        if sw_model==Poisson:
            model = sm.GLM(y_train, X_train, family=sw_model(link=sw_link)).fit()
        if sw_model==NegativeBinomial:
            model = sm.GLM(y_train, X_train, family=Poisson(link=sw_link)).fit()
            df_train_1 = pd.DataFrame({'BB_LAMBDA':model.mu, 'MC':y_train})    
            df_train_11 = pd.DataFrame({'AUX_OLS_DEP':df_train_1.apply(lambda x: ((x['MC'] - x['BB_LAMBDA'])**2 - x['BB_LAMBDA']) / x['BB_LAMBDA'], axis=1), 'BB_LAMBDA':model.mu, 'MC':y_train})    
            ols_expr = """AUX_OLS_DEP ~ BB_LAMBDA - 1"""     
            aux_olsr_results = smf.ols(ols_expr, df_train_11).fit() #Configure and fit the OLSR model:
            model = sm.GLM(y_train, X_train,family=sw_model(alpha=aux_olsr_results.params[0], link=sw_link)).fit()
        predictions = model.get_prediction(X_train)
        predictions_summary_frame = predictions.summary_frame()
        for t in np.arange(t_c,t_e):
            forecasts[t]= model.params[0]*X[t-max(lag[3],h),3] + model.params[1]*X[t-max(lag[0],h),0]+ model.params[2]*X[t-max(lag[2],h),2] + model.params[3]               
        forecasted_values=forecasts[t_c:t_e] 
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
        plt.plot(df.Date[t_i-max(lag[0],h):t_e-max(lag[0],h)], model.params[1]*X[t_i-max(lag[0],h):t_e-max(lag[0],h),0], color="green", label="Rainfall")
        plt.plot(df.Date[t_i-max(lag[2],h):t_e-max(lag[2],h)], model.params[2]*X[t_i-max(lag[2],h):t_e-max(lag[2],h),2], color="cyan", label="Humidity")
        plt.axhline(y=model.params[3], color="magenta", label="Intercept")
        plt.title("B", fontsize=20) 
        plt.ylabel("β_iX_j", fontsize=20)
        plt.xlabel("Date [months]", fontsize=20)
        plt.legend()
        plt.show()
    if var_add==['Malaria in the past', 'Rainfall', 'Temperature']: #baseline of explanatory variables
        X_train=np.c_[X[t_i-max(lag[3],h):t_c-max(lag[3],h),3], X[t_i-max(lag[0],h):t_c-max(lag[0],h),0],X[t_i-max(lag[1],h):t_c-max(lag[1],h),1], X[t_i-h:t_c-h,4]]
        if sw_model==Poisson:
            model = sm.GLM(y_train, X_train, family=sw_model(link=sw_link)).fit()
        if sw_model==NegativeBinomial:
            model = sm.GLM(y_train, X_train, family=Poisson(link=sw_link)).fit()
            df_train_1 = pd.DataFrame({'BB_LAMBDA':model.mu, 'MC':y_train})    
            df_train_11 = pd.DataFrame({'AUX_OLS_DEP':df_train_1.apply(lambda x: ((x['MC'] - x['BB_LAMBDA'])**2 - x['BB_LAMBDA']) / x['BB_LAMBDA'], axis=1), 'BB_LAMBDA':model.mu, 'MC':y_train})    
            ols_expr = """AUX_OLS_DEP ~ BB_LAMBDA - 1"""     
            aux_olsr_results = smf.ols(ols_expr, df_train_11).fit() #Configure and fit the OLSR model:
            model = sm.GLM(y_train, X_train,family=sw_model(alpha=aux_olsr_results.params[0], link=sw_link)).fit()
        predictions = model.get_prediction(X_train)
        predictions_summary_frame = predictions.summary_frame()
        for t in np.arange(t_c,t_e):
            forecasts[t]= model.params[0]*X[t-max(lag[3],h),3] + model.params[1]*X[t-max(lag[0],h),0] + model.params[2]*X[t-max(lag[1],h),1] + model.params[3]               
        forecasted_values=forecasts[t_c:t_e] 
        #We show the plots
        plt.figure(figsize=(16,4))
        plt.subplot(121)
        plt.plot(df.Date, y_o, color = "black" ,  marker='o', markersize=8)
        plt.plot(df.Date[t_i:t_c], model.mu, color = "blue", marker='o', markersize=8)
        plt.plot(df.Date[t_i:t_c], predictions_summary_frame.iloc[:,2], "y--", linewidth=2)
        plt.plot(df.Date[t_i:t_c], predictions_summary_frame.iloc[:,3], "g--", linewidth=2)
        plt.plot(df.Date[t_c:t_e], forecasted_values, color = "red", marker='o', markersize=8)
        plt.xlabel("Date [months]", fontsize=20)
        plt.title("A", fontsize=20)  
        plt.ylabel("Malaria incidence", fontsize=20)
        plt.legend(["Malaria incidence count", "predicted data", "mean_ci_lower", "mean_ci_upper", "forecasted data"])
        plt.subplot(122)     
        plt.plot(df.Date[t_i-max(lag[3],h):t_e-max(lag[3],h)], model.params[0]*X[t_i-max(lag[3],h):t_e-max(lag[3],h),3], color="black", label="Malaria in the past")
        plt.plot(df.Date[t_i-max(lag[0],h):t_e-max(lag[0],h)], model.params[1]*X[t_i-max(lag[0],h):t_e-max(lag[0],h),0], color="green", label="Rainfall")
        plt.plot(df.Date[t_i-max(lag[1],h):t_e-max(lag[1],h)], model.params[2]*X[t_i-max(lag[1],h):t_e-max(lag[1],h),1], color="red", label="Temperature")
        plt.axhline(y=model.params[3], color="magenta", label="Intercept")
        plt.title("B", fontsize=20) 
        plt.ylabel("β_iX_j", fontsize=20)
        plt.xlabel("Date [months]", fontsize=20)
        plt.legend()
        plt.show()
    if var_add==['Malaria in the past', 'Temperature', 'Humidity']: #baseline of explanatory variables
            X_train=np.c_[X[t_i-max(lag[3],h):t_c-max(lag[3],h),3], X[t_i-max(lag[1],h):t_c-max(lag[1],h),1],X[t_i-max(lag[2],h):t_c-max(lag[2],h),2], X[t_i-h:t_c-h,4]]
            if sw_model==Poisson:
                model = sm.GLM(y_train, X_train, family=sw_model(link=sw_link)).fit()
            if sw_model==NegativeBinomial:
                model = sm.GLM(y_train, X_train, family=Poisson(link=sw_link)).fit()
                df_train_1 = pd.DataFrame({'BB_LAMBDA':model.mu, 'MC':y_train})    
                df_train_11 = pd.DataFrame({'AUX_OLS_DEP':df_train_1.apply(lambda x: ((x['MC'] - x['BB_LAMBDA'])**2 - x['BB_LAMBDA']) / x['BB_LAMBDA'], axis=1), 'BB_LAMBDA':model.mu, 'MC':y_train})    
                ols_expr = """AUX_OLS_DEP ~ BB_LAMBDA - 1"""     
                aux_olsr_results = smf.ols(ols_expr, df_train_11).fit() #Configure and fit the OLSR model:
                model = sm.GLM(y_train, X_train,family=sw_model(alpha=aux_olsr_results.params[0], link=sw_link)).fit()
            predictions = model.get_prediction(X_train)
            predictions_summary_frame = predictions.summary_frame()
            for t in np.arange(t_c,t_e):
                forecasts[t]= model.params[0]*X[t-max(lag[3],h),3] + model.params[1]*X[t-max(lag[1],h),1]+ model.params[2]*X[t-max(lag[2],h),2] + model.params[3]              
            forecasted_values=forecasts[t_c:t_e] 
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
            plt.plot(df.Date[t_i-max(lag[1],h):t_e-max(lag[1],h)], model.params[1]*X[t_i-max(lag[1],h):t_e-max(lag[1],h),1], color="red", label="Temperature")
            plt.plot(df.Date[t_i-max(lag[2],h):t_e-max(lag[2],h)], model.params[2]*X[t_i-max(lag[2],h):t_e-max(lag[2],h),2], color="cyan", label="Humidity")
            plt.axhline(y=model.params[3], color="magenta", label="Intercept")
            plt.title("B", fontsize=20) 
            plt.ylabel("β_iX_j", fontsize=20)
            plt.xlabel("Date [months]", fontsize=20)
            plt.legend()
            plt.show()
    if var_add==['Rainfall', 'Temperature', 'Humidity', 'Malaria_in_the_past']:
        X_train=np.c_[X[t_i-max(lag[0],h):t_c-max(lag[0],h),0], X[t_i-max(lag[1],h):t_c-max(lag[1],h),1], X[t_i-max(lag[2],h):t_c-max(lag[2],h),2], X[t_i-max(lag[3],h):t_c-max(lag[3],h),3], X[t_i-h:t_c-h,4]]
        if sw_model==Poisson:
            model = sm.GLM(y_train, X_train, family=sw_model(link=sw_link)).fit()
        if sw_model==NegativeBinomial:
            model = sm.GLM(y_train, X_train, family=Poisson(link=sw_link)).fit()
            df_train_1 = pd.DataFrame({'BB_LAMBDA':model.mu, 'MC':y_train})    
            df_train_11 = pd.DataFrame({'AUX_OLS_DEP':df_train_1.apply(lambda x: ((x['MC'] - x['BB_LAMBDA'])**2 - x['MC']) / x['BB_LAMBDA'], axis=1), 'BB_LAMBDA':model.mu, 'MC':y_train})    
            ols_expr = """AUX_OLS_DEP ~ BB_LAMBDA - 1"""     
            aux_olsr_results = smf.ols(ols_expr, df_train_11).fit() #Configure and fit the OLSR model:
            model = sm.GLM(y_train, X_train,family=sw_model(alpha=aux_olsr_results.params[0], link=sw_link)).fit() 
        predictions = model.get_prediction(X_train)
        predictions_summary_frame = predictions.summary_frame()
        for t in np.arange(t_c,t_e):
            forecasts[t]= model.params[0]*X[t-max(lag[0],h),0] + model.params[1]*X[t-max(lag[1],h),1]  + model.params[2]*X[t-max(lag[2],h),2] + model.params[3]*X[t-max(lag[3],h),3] + model.params[4]                
        forecasted_values=forecasts[t_c:t_e] 
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
        plt.plot(df.Date[t_i-max(lag[0],h):t_e-max(lag[0],h)], model.params[0]*X[t_i-max(lag[0],h):t_e-max(lag[0],h),0], color="green", label="Rainfall")
        plt.plot(df.Date[t_i-max(lag[1],h):t_e-max(lag[1],h)], model.params[1]*X[t_i-max(lag[1],h):t_e-max(lag[1],h),1], color="red", label="Temperature") 
        plt.plot(df.Date[t_i-max(lag[2],h):t_e-max(lag[2],h)], model.params[2]*X[t_i-max(lag[2],h):t_e-max(lag[2],h),2], color="cyan", label="Humidity") 
        plt.plot(df.Date[t_i-max(lag[3],h):t_e-max(lag[3],h)], model.params[3]*X[t_i-max(lag[3],h):t_e-max(lag[3],h),3], color="black", label="Malaria in the past") 
        plt.axhline(y=model.params[4], color="magenta", label="Intercept")
        plt.title("B",  fontsize=20)
        plt.ylabel("β_iX_j", fontsize=20)
        plt.xlabel("Date [months]", fontsize=20)
        plt.legend()
        plt.show()  
       
    print("******************** FORECAST ACCURACY MEASURES - latex table ********************")
    print('\\begin{tabular}{|c|c|c|c|c|c|c|c|}')
    print('\\hline')
    #print("RMSE_train",'&',"RMSE_test",'&',  "MASE_train",'&',"MASE_test",'&', "MARE_train",'&',"MARE_test",'&', "R2_train",'&',"R2_test",'&', "Min",'\\\\')
    print('Regions & Model & Links & RMSE\_train/RMSE_test & MASE\_train/MASE\_test & MARE\_train/MARE\_test & R2\_train/R2\_test & Min \\\\')
    print('\\hline')
    accuracy_measures(sw_regions, sw_model, sw_link, 'each_link', model.mu, forecasted_values, y_train, y_test, t_i, t_c, t_e, y_o, min(model.mu))
    print('\\end{tabular}')
    return

#********** Calling the function with the following input *******
    #sw_regions = ['Dakar', 'Fatick', 'Kedougou']
    #var_add=[['Malaria in the past'],['Malaria in the past', 'Rainfall'],['Malaria in the past', 'Temperature'],['Malaria in the past', 'Humidity'],['Malaria in the past','Rainfall','Temperature'],['Malaria in the past','Rainfall','Humidity'],['Malaria in the past','Temperature','Humidity'],['Rainfall', 'Temperature', 'Humidity', 'Malaria_in_the_past']]
    #sw_model = [Poisson, NegativeBinomial]
addition_study(sw_regions = 'Dakar',  
                var_add=['Malaria in the past'],
                sw_model = Poisson
                )                    
         
                      
    