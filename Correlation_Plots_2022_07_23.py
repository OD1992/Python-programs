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

t_i=6; t_e=108
print("******************** Correlations - latex table ********************")
for sw_regions in ['Dakar', 'Fatick', 'Kedougou']:
    #j=['Dakar', 'Fatick', 'Kedougou'].index(sw_regions)
    print('\\begin{tabular}{c|ccc}')
    print('\\hline')
    print(' & Variables correlated & Correlation Values \\\\ \\hline')
    if sw_regions == 'Dakar':  
        y = df.MC_Dakar.values[t_i:t_e]
        X = np.c_[df.Rainfall, AVT_Dakar, df.Humidity_D, df.Milda_D, df.ACT_D, df.MC_Dakar]
    if sw_regions == 'Fatick':
        y = df.MC_Fatick.values[t_i:t_e]
        X = np.c_[df.Rainfall_F, AVT_Fatick, df.Humidity_F, df.Milda_F, df.ACT_F,df.MC_Fatick]      
    if sw_regions == 'Kedougou':
        y = df.MC_Kedougou.values[t_i:t_e]
        X = np.c_[df.Rainfall_K, AVT_Kedougou, df.Humidity_K, df.Milda_K, df.ACT_K, df.MC_Kedougou]

    matrice = pd.DataFrame({'Rainfall':X[:,0], 'Temperature':X[:,1], 'Humidity':X[:,2], 'Milda':X[:,3], 'ACT':X[:,4], 'MC past':X[:,5]}) 
    fig=plt.figure(figsize=(10,6))
    for i in np.arange(0,np.shape(X)[1]):
        lag_layout=np.full(np.shape(X)[1], np.nan)
        correlation=np.full(t_i, np.nan)
        for lag in np.arange(0,t_i):
            correlation[lag]=np.corrcoef(y, X[t_i-lag:t_e-lag,i])[0][1]
        p=np.argmax(np.abs(correlation))
        print('y_o(t)\&',matrice.columns[i] ,'(t-',p,')',"&",round(max(np.abs(correlation)),2), '\\\\')
        plt.plot(np.arange(0,t_i), correlation, label="r(MC,"+str(matrice.columns[i])+")", marker='o', markersize=8)
    
    print('\\end{tabular}') 
    plt.ylim(-1,1)
    plt.xlabel("lag", fontsize=20)
    plt.ylabel("Correlation values", fontsize=20)
    #plt.title("Region: "+str(sw_regions), fontsize=20)
    plt.legend(fontsize=15)
    plt.show() 
    #fig.savefig('C:/Users/odiao/Dropbox/Model paludisme/figures_palu_rapport/correlation_kd_2022_01_14.eps')   # save the figure to file
    #plt.close(fig)