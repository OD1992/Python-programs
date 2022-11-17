# -*- coding: utf-8 -*-
"""
Created on Sat Jul 23 12:33:18 2022

@author: odiao
"""
import warnings
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.genmod.families import Poisson, NegativeBinomial
from statsmodels.genmod.families.links import identity, log, sqrt
from scipy import stats




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
#******************The vuong test between Poisson and NB*************************************
#Vunog test function
t_i=5; t_c=84; t_e=108; intercept=np.ones(t_e); h=1 
print("******************** FORECAST ACCURACY MEASURES - latex table ********************")
print('\\begin{tabular}{|c|c|c|}')
print('\\hline')
print('Regions & Link & V \\\\ \\hline')
for sw_regions in ['Dakar', 'Fatick', 'Kedougou']:
    for sw_link in [identity, log, sqrt]:
        if sw_regions == 'Dakar':  
            lag=[2,2,5,1] # if we applied the restriction: lag>=2 in rainfall, temperature and humidity but lag stays equal to 1 for malaria in the past.
            y_o = df.MC_Dakar.values
            X = np.c_[df.Rainfall, AVT_Dakar, df.Humidity_D, df.MC_Dakar, intercept]
        if sw_regions == 'Fatick':
            lag=[3,4,3,1] # if we applied the restriction: lag>=2
            y_o = df.MC_Fatick.values
            X = np.c_[df.Rainfall_F, AVT_Fatick, df.Humidity_F, df.MC_Fatick, intercept]      
        if sw_regions == 'Kedougou':
            lag=[2,5,2,1] # if we applied the restriction: lag>=2
            y_o = df.MC_Kedougou.values
            X = np.c_[df.Rainfall_K, AVT_Kedougou, df.Humidity_K, df.MC_Kedougou, intercept]
        y_train=y_o[t_i:t_c]  
        X_train = np.c_[X[t_i-max(h,lag[0]):t_c-max(h,lag[0]),0], X[t_i-max(h,lag[1]):t_c-max(h,lag[1]),1], X[t_i-max(h,lag[2]):t_c-max(h,lag[2]),2], X[t_i-max(h,lag[3]):t_c-max(h,lag[3]),3], X[t_i-h:t_c-h,4]]
        #**************Poisson***********************************
        if sw_link==identity:
            model_P = sm.GLM(y_train, X_train, family=Poisson(link=sw_link)).fit()        
        if sw_link==log:
            model_P = sm.GLM(y_train, X_train, family=Poisson(link=sw_link)).fit()
        if sw_link==sqrt:
            model_P = sm.GLM(y_train, X_train, family=Poisson(link=sw_link)).fit()
        poisson_mass_function=stats.poisson.pmf(y_train,  model_P.mu)
        poisson_mass_function=poisson_mass_function[poisson_mass_function>0.0]
        #For GLM NB model, we first fit the GLM Poisson model in order to obtain the lamda's which permit to use the OLS optim moethod in order to determine the dispersion paramter
        #****************NegativeBinomial*****************************
        base1 = pd.DataFrame({'MC':y_train, 'LAMBDA':model_P.mu})
        base_ols = pd.DataFrame({'AUX_OLS_DEP':base1.apply(lambda x: ((x['MC'] - x['LAMBDA'])**2 - x['LAMBDA']) / x['LAMBDA'], axis=1),'LAMBDA':model_P.mu, 'MC':y_train})
        results = sm.OLS(base_ols[['AUX_OLS_DEP']], base_ols[['LAMBDA']]).fit() #Configure and fit the OLSR model
        model_nb = sm.GLM(y_train, X_train,family=NegativeBinomial(alpha=results.params[0], link=sw_link)).fit()
        nb_mass_function=stats.nbinom.pmf(y_train, model_nb.mu, p=results.params[0])
        nb_mass_function=nb_mass_function[nb_mass_function>0]
        if len(poisson_mass_function)<len(nb_mass_function):
            frac=poisson_mass_function/nb_mass_function[:len(poisson_mass_function)] #To avoid to devide by zero in the numerator
        else:
            frac=poisson_mass_function[:len(nb_mass_function)]/nb_mass_function
        m_i=np.log(frac)
        m_i=list(filter(float('inf').__ne__, m_i)) # To filter the value=inf
        V=np.sqrt(len(m_i))*np.mean(m_i)/np.sqrt(np.var(m_i))
        if sw_link==identity:
            sw_link='id'
        if sw_link==log:
            sw_link='log'
        if sw_link==sqrt:
            sw_link='sqrt'
        print(sw_regions,"&",sw_link, "&", round(V,2), '\\\\')
print('\\hline')
print('\\end{tabular}') 
    