from statsmodels.tsa.arima_model import ARIMA
import  warnings 
#from pandas.plotting import autocorrelation_plot
import pandas as pd
import numpy as np
import copy
from datetime import datetime   # useful for date ranges in plots
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('xtick', labelsize=40) 
matplotlib.rc('ytick', labelsize=40) 
import seaborn as sns
sns.set(style="whitegrid", color_codes=True)
from sklearn.metrics import mean_squared_error
#ACF and PACF plots:
from statsmodels.tsa.stattools import acf, pacf
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
data=total_in[:-20]
#autocorrelation_plot(total_in)

def MAPE(y_true, y_pred):  # MAPE is "Mean Absolute Percentage Error"
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    

#To make stationary the series
plt.plot(np.diff(total_in,1))
plt.plot(np.diff(total_in,2))

#**********************************************************************************
#Forecasting values and plots
train, test = total_in[0:30], total_in[30:]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
    #The forecast() function performs a one-step forecast using the model
    #A rolling forecast is required given the dependence on observations in prior time steps for differencing and the AR model.
    model = ARIMA(history, order=(5,1,0)) #WE use a difference order of 1 to make the time series stationary
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
	#print('predicted=%f, expected=%f' % (yhat, obs))
error = np.sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % error)
# plot
plt.figure(figsize = [12, 6])
#print("Predictions with arima (5,2,0)")
plt.plot(dates, total_in, color="gray", label="Data")
plt.plot(dates[0:30], train, 'b.-', label="Train")
plt.plot(dates[30:], predictions, 'r--', label="Predict")
plt.legend()

residuals = pd.DataFrame(model_fit.resid)
residuals.plot()

plt.show()       

#****************************************************
# Forecasting by changing train period
plt.figure(figsize = [12, 6])
print("Predictions with arima (5,2,0)")
plt.plot(dates, total_in, color="gray")

for period in [20,30,40,50,60,70]:
    train, test = total_in[0:period], total_in[period:]
    history = [x for x in train]
    predictions = list()
    for t in range(len(test)):
        #A rolling forecast is required given the dependence on observations in prior time steps for differencing and the AR model.
        model = ARIMA(history, order=(5,1,0)) 
        model_fit = model.fit(disp=0)
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test[t]
        history.append(obs)
    	#print('predicted=%f, expected=%f' % (yhat, obs))
    error = np.sqrt(mean_squared_error(test, predictions))
    print('Test RMSE: %.3f' % error)
    plt.plot(dates[0:period], train, 'b--')
    plt.plot(dates[period:], predictions, 'r--')
    

plt.legend(["Data","Train","Predict"])
plt.show()       




#*********************************************************
#Forecasting values
plt.figure()#figsize = [10, 6])
x=np.arange(0,len(dates))
predi= list()
train= total_in[0:40]

model = ARIMA(train, order=(5,2,0))
result= model.fit()
output = result.forecast(steps=30) 
yhat = output[0]
predi.append(yhat)
print(predi)

plt.plot(x, total_in, color='grey', lw=3, label="Observed")
plt.plot(x[:len(train)], total_in[:len(train)], 'b--', lw=2, label="Train")
plt.plot(x[len(train):70],predi[0], 'r--') 
#plt.legend(["Observed_values","Training_values", "Predicted"], loc='best')
plt.xlabel('Time /days',size=10)
plt.ylabel('Hospitalized',size=10)
plt.show()



from pandas.plotting import autocorrelation_plot
autocorrelation_plot(total_in)
plt.show()
#******************************************************
fig=plt.figure(figsize=(12, 7))
plt.subplot(121)
lag_acf = acf(total_in)#, nlags=50)
#Plot ACF: 
plt.plot(lag_acf, marker="o")
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(total_in)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(total_in)),linestyle='--',color='gray')
plt.title('Autocorrelation Function')
plt.xlabel('number of lags')
plt.ylabel('correlation')
plt.tight_layout()
#calling pacf function from stattool and PLOT PACF
plt.subplot(122)
lag_pacf = pacf(total_in, method='ols')
plt.plot(lag_pacf, marker="o")
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(total_in)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(total_in)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.xlabel('number of lags')
plt.ylabel('correlation')
plt.tight_layout()
#fig.savefig('C:/Users/odiao/Desktop/Redaction_darticles_Latex/SHR_PA/Figures/acf_pacf.pdf')   # save the figure to file
#plt.close(fig)
