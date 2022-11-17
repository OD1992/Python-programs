import numpy as np
import math
import pandas as pd
import copy
from datetime import datetime   # useful for date ranges in plots
import matplotlib.pyplot as plt
import os
os.chdir ('C:\\Users\\odiao\\Desktop\\Model Covid19\\programme_python\\SHR_PA\Code_Python')
os.getcwd ()

data_raw = pd.read_csv('Data/Belgium/COVID19BE_HOSP_2020-07-16.csv')
data_raw_state = data_raw.groupby('DATE', as_index=False).sum()  # sum over provinces
dataa = data_raw_state[['DATE', 'NR_REPORTING', 'TOTAL_IN','TOTAL_IN_ICU','TOTAL_IN_RESP','TOTAL_IN_ECMO','NEW_IN','NEW_OUT']]  # exclude some useless columns
        
# Extract relevant data and recompute new_out:
# Source: Some variable names taken from https://rpubs.com/JMBodart/Covid19-hosp-be
data_length = np.size(dataa,0)
data_num = dataa.iloc[:,1:].to_numpy(dtype=float)  # extract all rows and 2nd-last rows (recall that Python uses 0-based indexing) and turn it into a numpy array of flats. The "float" type is crucial due to the use of np.nan below. (Setting an integer to np.nan does not do what it is should do.)

#dates = data['DATE'])
dates_raw = copy.deepcopy(dataa['DATE'])
dates_raw = dates_raw.reset_index(drop=True)  # otherwise the index is not contiguous when sw_states = 'each'
dates = [None] * data_length
for i in range(0,data_length):
    dates[i] = datetime.strptime(dates_raw[i],'%Y-%m-%d')

col_total_in = 1
total_in = data_num[:,col_total_in]
 # Define functions for statistics:
def MAPE(y_true, y_pred):  # MAPE is "Mean Absolute Percentage Error"
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
print("******************** FORECAST ACCURACY MEASURES - latex table ********************")
def make_latex_table(stats_all):
    print('\\begin{tabular}{c|c|c}')
    print('\\hline')
    print('Train Interval & MAPE train & MAPE_test \\\\ \\hline')
    #for key in stats_keys_for_latex:
    print(val, '&', "{:.2f}".format(MAPE_train), '&', "{:.2f}".format(MAPE_test), '\\\\')
        #if '/' in key:
    print('\\hline')
    print('\\end{tabular}')
    
x=np.arange(0,len(dates))
y=np.array(total_in)

for val in [20,40,60,80,100]:
    x=np.arange(0,len(dates))
    train_set=total_in[:val]
    test_set=total_in[val:]
    #fitting the model in training period
    x_train=x[0:val]
    y_train=np.array(train_set)
    P = np.poly1d(np.polyfit(x_train, y_train, 15))
   # print(P)
    #XP_train = np.linspace(0, len(train_set), 100)
    MAPE_train = MAPE(train_set, P(x[:val]))
    #XP_test = np.linspace(len(train_set), len(total_in), 100)
    #x_test= np.arange(len(train_set),len(total_in))
    MAPE_test = MAPE(test_set, P(x[val:]))
    stats_all=val, MAPE_train, MAPE_test
    make_latex_table(stats_all)
 
    
#fitting model in whole data 
fig=plt.figure(figsize = [16, 10])
plt.subplot(211)
x=np.arange(0,len(dates))
y=np.array(total_in)
p = np.poly1d(np.polyfit(x, y, 15))
print(p)
xp = np.linspace(0, len(total_in), 100)
plt.plot(x, total_in, "-",color='gray', lw=3, label="Observed")
plt.plot(xp, p(xp), 'b', lw=3,label="Predicted")
plt.ylabel('Hospitalization cases',size=20)  
plt.title("Fitting polynomial model in whole data", size=20)
plt.legend() 
#Plot

plt.subplot(212)
plt.plot(x, total_in, '--',color='grey', lw=1)
plt.xlabel('Time /days',size=15)
plt.ylabel('Hospitalization cases',size=15)
plt.ylim(-1e4, 1e4, 1e2)
for val in [20,40,60,80,100]:
    x=np.arange(0,len(dates))
    train_set=total_in[:len(total_in)-val]
    test_set=total_in[len(total_in)-val:]
    #fitting the model in training period
    x_train=np.arange(0,len(train_set))
    y_train=np.array(train_set)
    P = np.poly1d(np.polyfit(x_train, y_train, 15))
   # print(P)
    XP_train = np.linspace(0, len(train_set), 100)
    
    XP_test = np.linspace(len(train_set), len(total_in), 100)
    x_test= np.arange(len(train_set),len(total_in))
    #plt.plot(x_train, P(x_train), 'b--',lw=3)
    plt.plot(x[:val], total_in[:val], 'b--',lw=3)
    #plt.plot(x_test, P(x_test), 'r-.', lw=3)  
    plt.plot(x[val:], P(x[val:]), 'r-.', lw=3) 
plt.legend(["Observed","Training_values", "Predicted_values"])
plt.show()

fig.savefig('C:/Users/odiao/Desktop/Redaction_darticles_Latex/SHR_PA/Figures/poly.eps')   # save the figure to file
plt.close(fig)    # close the figure window



