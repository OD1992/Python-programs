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
    print('\\begin{tabular}{c|c}')
    print('\\hline')
    print('Train Interval & MAPE_test \\\\ \\hline')
    #for key in stats_keys_for_latex:
    print([t_i,  t_c ], '&', "{:.2f}".format(MAPE_test), '\\\\')
        #if '/' in key:
    print('\\hline')
    print('\\end{tabular}')
    
    
#def expo(t):
t_i_vals=np.arange(0, len(total_in)-30,5)

#Stats
for t_i in t_i_vals:
    t_c = t_i + 15
    H = np.full(len(total_in), np.nan)  # set storage
    H[t_i] = total_in[t_i]
    tau=-(t_c-t_i)/(np.log(total_in[t_c])-np.log(total_in[t_i]))
    for t in np.arange(t_c,len(total_in)):
        H[t] = H[t_i]*math.exp(-(t-t_i)/tau)  
    #print(H)
    #print("training period:", [t_i,  t_c ])
    MAPE_test = MAPE(total_in[t_c:len(total_in)], H[t_c:len(total_in)])
    #print("MAPE_test:", MAPE_test)
    stats_all=[t_i,  t_c ], MAPE_test
    make_latex_table(stats_all)


#Plot
fig=plt.figure(figsize = [10, 6])
plt.plot(dates, total_in, color='grey', lw=3)
plt.xlabel('Time /days',size=15)
plt.ylabel('Hospitalization cases',size=15)
plt.ylim(0,7000,100)
t_i_vals=np.arange(0, len(total_in)-30,5)
for t_i in t_i_vals:
    t_c = t_i + 15
    H = np.full(len(total_in), np.nan)  # set storage
    H[t_i] = total_in[t_i]
    tau=-(t_c-t_i)/(np.log(total_in[t_c])-np.log(total_in[t_i]))
    for t in np.arange(t_c,len(total_in)):
        H[t] = H[t_i]*math.exp(-(t-t_i)/tau)  
    #print(H)
    plt.plot(dates[t_i:t_c],  total_in[t_i:t_c], 'b--', lw=3)
    plt.plot(dates, H, 'r-.', lw=3)
    
plt.legend(['Obeserved_hospi_data', 'Training_values', 'Predictied_values'])
plt.show()

fig.savefig('C:/Users/odiao/Desktop/Redaction_darticles_Latex/SHR_PA/Figures/expo.eps')   # save the figure to file
plt.close(fig)    # close the figure window












