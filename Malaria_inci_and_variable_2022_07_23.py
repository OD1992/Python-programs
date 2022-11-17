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

Vect=[(df['MC_Dakar'], df['Rainfall']), (df['MC_Fatick'], df['Rainfall_F']), (df['MC_Kedougou'], df['Rainfall_K']), 
      (df['MC_Dakar'], df['Milda_D']), (df['MC_Fatick'], df['Milda_F']), (df['MC_Kedougou'], df['Milda_K'])]

for p in np.arange(0, len(Vect)):
    fig, ax1=plt.subplots(figsize=(7,3))
    ax2=ax1.twinx()
    ax1.bar(df.Date, Vect[p][0], width= 20, align='center', color = "black" )
    if p<3:
        ax2.plot(df.Date, Vect[p][1], color = "green", marker='o', markersize=8)
    else:
        ax2.plot(df.Date, Vect[p][1], color = "blue", marker='o', markersize=8)
    ax1.set_ylabel("Incidence", fontsize=20) 
    ax2.set_ylabel("Rainfall", color='green', fontsize=20) 
    ax2.set_xlabel("Date [months]", fontsize=20)
    plt.show() 