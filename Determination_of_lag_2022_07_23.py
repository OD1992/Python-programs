# -*- coding: utf-8 -*-
"""
Created on Sat Jul 23 12:33:18 2022

@author: odiao
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os
#os.chdir ('C:\\Users\\odiao\\Desktop\\Model paludisme')
os.chdir('C:\\Users\\odiao\\Dropbox\\Model paludisme')
df = pd.read_csv('Dakar.csv', header=0, infer_datetime_format=False, parse_dates=[0])

#***************************************************************************************************
#Defined the avaerage temperature
AVT_Dakar=(df.Tempmin_Dakar + df.Tempmax_Dakar)/2
AVT_Fatick=(df.Tempmin_Fatick + df.Tempmax_Fatick)/2
AVT_Kedougou=(df.Tempmin_Kedougou + df.Tempmax_Kedougou)/2

#*********************************************************************************************************************
#Correlation between the observed malaria cases and the explanatory variables in different lag to find the best values
#With the restriction: lag>=2
t_i=7; t_e=108
for sw_regions in ['Dakar', 'Fatick', 'Kedougou']:
    if sw_regions == 'Dakar':  
        y = df.MC_Dakar.values[t_i:t_e]
        X = np.c_[df.Rainfall, AVT_Dakar, df.Humidity_D, df.MC_Dakar]
    if sw_regions == 'Fatick':
        y = df.MC_Fatick.values[t_i:t_e]
        X = np.c_[df.Rainfall_F, AVT_Fatick, df.Humidity_F, df.MC_Fatick]      
    if sw_regions == 'Kedougou':
        y = df.MC_Kedougou.values[t_i:t_e]
        X = np.c_[df.Rainfall_K, AVT_Kedougou, df.Humidity_K, df.MC_Kedougou]

    matrice = pd.DataFrame({'Rainfall':X[:,0], 'Temperature':X[:,1], 'Humidity':X[:,2], 'MC past':X[:,3]}) 
    fig=plt.figure(figsize=(6,3))
    for i in np.arange(0,np.shape(matrice)[1]):
        correlation=np.full(t_i, np.nan)
        for lag in np.arange(0,t_i):
            correlation[lag]=np.corrcoef(y, matrice.iloc[t_i-lag:t_e-lag,i])[0][1]
        print(correlation)
        print("If lag>=2, [max, index]= ", [max(np.abs(correlation[2:])), 2+np.argmax(np.abs(correlation[2:]))])
        plt.plot(np.arange(0,t_i), correlation, label="Correlation(MC,"+str(matrice.columns[i])+")", marker='o', markersize=8)
    plt.ylim(-1,1)
    plt.xlabel("lag", fontsize=20)
    plt.ylabel("Correlation values", fontsize=20)
    plt.title("Region: "+str(sw_regions), fontsize=20)
    plt.legend(fontsize=12)
    plt.show()  
    