import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math 
#from mpl_toolkits.mplot3d import Axes3D
#from math import hypot
from scipy import optimize
#from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
import copy
from datetime import datetime   # useful for date ranges in plots
#To change repertory
import os
os.chdir ('C:\\Users\\odiao\\Desktop\\Model Covid19\\programme_python\\SHR_PA\Code_Python')
os.getcwd ()
# ***********************************************************************************
# Switches and other user choices - see also sw_periods and c_H, c_E, c_L below
# *******
sw_dataset = 'BEL'  # !! Default: 'BEL'. Currently 'BEL' and 'FRA' 'LUX'and 'UK'are available.
sw_districts = 'sum'  # !! Default: 'sum'. If 'sum', sums over all districts (i.e., provinces, departments...). If 'each', loop over all districts. sw_districts can also be the name of a district (for example, sw_districts = 'Brussels', or sw_districts = '75' for Paris).

show_totinout = 1  # Default: 0 # If 1, shows plots of total, in and out and check their discrepancy.
save_figures = 0  # If 1, some figures will be saved in pdf format.

show_figures = 1  # If 0, no figure shown. Will be set to 0 later on if nb_districts too large.
show_hist = 0
show_H = 1  # If 1, draw a plot of the evolution of H(t).
show_S_bar = 1
show_beta_bar = 1
show_gamma = 1
show_mu = 1
show_L = 1
show_D_by_day = 1
# ***********************************************************************************
# Load data Belgium
# *******
if sw_dataset == 'BEL':   
    # The data comes from https://epistat.sciensano.be/Data/COVID19BE_HOSP.csv.
    # This link was provided on 22 June 2020 by Alexey Medvedev on the "Re R0 estimation" channel of the O365G-covidata team on MS-Teams.
    # The link can be reached from https://epistat.wiv-isp.be/covid/
    # Some explanations can be found at https://epistat.sciensano.be/COVID19BE_codebook.pdf

    data_raw = pd.read_csv('Data/Belgium/COVID19BE_HOSP_2020-07-16.csv')
    data_death = pd.read_csv('Data/Belgium/COVID19BE_DEATH_2020-07-16.csv')
    #fields = ['DATE', 'NR_REPORTING', 'TOTAL_IN','TOTAL_IN_ICU','TOTAL_IN_RESP','TOTAL_IN_ECMO','NEW_IN','NEW_OUT']

    if sw_districts == 'each':
        data_groupbydistrict = pd.DataFrame(data_raw.groupby("PROVINCE"))
# ***********************************************************************************
# Load data France
# *******

if sw_dataset == 'FRA':
    # The data comes from https://www.data.gouv.fr/en/datasets/donnees-hospitalieres-relatives-a-lepidemie-de-covid-19/, see donnees-hospitalieres-covid19-2020-07-10-19h00.csv

    data_raw = pd.read_csv('Data/France/donnees-hospitalieres-covid19-2020-07-17-19h00_corrected.csv')  # some dates were not in the correct format, hence the "corrected" version of the csv file
    data_raw = data_raw[data_raw.iloc[:,1]==0].reset_index(drop=True)  # Discard sex "1" and "2" (i.e., only keep sex "0" which is the sum of females and males) and reset the index in order to have a contiguous index in the DataFrame.

    if sw_districts == 'each':
        data_groupbydistrict = pd.DataFrame(data_raw.groupby("dep"))
# ***********************************************************************************
# Load data Luxembourg
# *******
if sw_dataset == 'LUX':
    # The data comes from https://www.data.gouv.fr/en/datasets/donnees-hospitalieres-relatives-a-lepidemie-de-covid-19/, see donnees-hospitalieres-covid19-2020-07-10-19h00.csv

    data_raw = pd.read_csv('Data/Luxembourg/data_lux.csv')  # some dates were not in the correct format, hence the "corrected" version of the csv file
    #data_raw = data_raw[data_raw.iloc[:,1]==0].reset_index(drop=True)  # Discard sex "1" and "2" (i.e., only keep sex "0" which is the sum of females and males) and reset the index in order to have a contiguous index in the DataFrame.
    data_groupbydistrict=pd.DataFrame(data_raw.groupby("Date"))
# ***********************************************************************************
# Load data United kingdom
# *******
if sw_dataset == 'UK':
    # The data comes from https://www.data.gouv.fr/en/datasets/donnees-hospitalieres-relatives-a-lepidemie-de-covid-19/, see donnees-hospitalieres-covid19-2020-07-10-19h00.csv

    data_raw = pd.read_csv('Data/Uk/data_UK.csv')  # some dates were not in the correct format, hence the "corrected" version of the csv file
    #data_raw = data_raw[data_raw.iloc[:,1]==0].reset_index(drop=True)  # Discard sex "1" and "2" (i.e., only keep sex "0" which is the sum of females and males) and reset the index in order to have a contiguous index in the DataFrame.
    data_groupbydistrict=pd.DataFrame(data_raw.groupby("date"))
# {{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{
# Start loop on districts
# *******      
if sw_districts == 'sum':
    nb_districts = 1
elif sw_districts == 'each':
    nb_districts = len(data_groupbydistrict)
else:  # else nb_districts is the name of a district
    nb_districts = 1
    
if nb_districts > 2:
    show_figures = 0  # Force figures off if there are too many districts

for cnt_district in range(nb_districts):

    if sw_districts == 'sum':
        district_name = 'sum'
        district_names = np.array(['sum'])   # without np.array, we get an error in district_names[medians_argsort]
    elif sw_districts == 'each':
        district_name = data_groupbydistrict[0][cnt_district]
        district_names = data_groupbydistrict[0]
    else:
        district_name = sw_districts
        district_names = np.array([sw_districts])

    # ***********************************************************************************
    # Process data Belgium
    # *******
    if sw_dataset == 'BEL':

        if sw_districts == 'sum':
            data_raw_district = data_raw.groupby('DATE', as_index=False).sum()  # sum over provinces
        elif sw_districts == 'each':
            data_raw_district = data_groupbydistrict[1][cnt_district]  # extract province cnt_district
        else:   
            data_raw_district = data_raw[data_raw.iloc[:,1]==sw_districts].reset_index(drop=True)   # extract district with name sw_districts

        data = data_raw_district[['DATE', 'NR_REPORTING', 'TOTAL_IN','TOTAL_IN_ICU','TOTAL_IN_RESP','TOTAL_IN_ECMO','NEW_IN','NEW_OUT']]  # exclude some useless columns
            
        # Extract relevant data and recompute new_out:
        # Source: Some variable names taken from https://rpubs.com/JMBodart/Covid19-hosp-be
        data_length = np.size(data,0)
        data_num = data.iloc[:,1:].to_numpy(dtype=float)  # extract all rows and 2nd-last rows (recall that Python uses 0-based indexing) and turn it into a numpy array of flats. The "float" type is crucial due to the use of np.nan below. (Setting an integer to np.nan does not do what it is should do.)

        #dates = data['DATE'])
        dates_raw = copy.deepcopy(data['DATE'])
        dates_raw = dates_raw.reset_index(drop=True)  # otherwise the index is not contiguous when sw_districts = 'each'
        dates = [None] * data_length
        for i in range(0,data_length):
            dates[i] = datetime.strptime(dates_raw[i],'%Y-%m-%d')

        col_total_in = 1
        col_total_in_ICU=2
        col_total_in_RESP=3
        col_total_in_ECMO=4
        col_new_in = 5
        col_new_out = 6
        total_in = data_num[:,col_total_in]
        total_in_ICU=data_num[:,col_total_in_ICU]
        total_in_RESP=data_num[:,col_total_in_RESP]
        total_in_ECMO=data_num[:,col_total_in_ECMO]
        new_in = data_num[:,col_new_in]
        new_out_raw = data_num[:,col_new_out] # there will be a non-raw due to the "Problem" mentioned below.
        #For deaths
        death=data_death.iloc[:,1]
        new_delta = new_in - new_out_raw
        
        cum_new_delta = np.cumsum(new_delta)
        
        Critical_cases=total_in_RESP + total_in_ECMO
        total_in_chg = np.hstack(([0],np.diff(total_in))) #difference between x[i+1]-x[i]
        # Problem: new_delta and total_in_chg are different, though they are sometimes close. 
        # Cum_new_delta does not go back to something close to zero, whereas it should. Hence I should not trust it.
        # I'm going to trust total_in and new_in. I deduce new_out_fixed by:
        new_out = new_in - total_in_chg   # fixed new_out
        data_totinout = np.c_[total_in,new_in,new_out,death, new_out_raw,total_in_ICU, total_in_RESP, total_in_ECMO,Critical_cases]  # store total_in, new_in, new_out and death in an arraw with 4 columns
        
        nb_xticks = 4
        dates_ticks = [None] * nb_xticks
        dates_ticks_ind = np.linspace(0,len(total_in)-1,nb_xticks,dtype=int)
        for i in range(0,nb_xticks):
            dates_ticks[i] = dates[dates_ticks_ind[i]]
        fig=plt.figure(figsize=(14,8))
        plt.plot(dates,death, color='darkred',lw=3,label="Deaths")
        plt.plot(dates,new_out,color='cyan',lw=2,label="Leaving hospi_calculated")
        plt.plot(dates,new_out_raw, color='darkgreen', lw=3,label="Leaved hospital")
        plt.legend(fontsize=20)
        plt.ylabel("Values", fontsize=20)
        plt.xlabel("Dates", fontsize=20)
        plt.xticks(dates_ticks, fontsize=15)
        plt.yticks(fontsize=15)
        plt.show()
        #fig.savefig('C:/Users/odiao/Desktop/Model Covid19/programme_python/SHR_PA/Code_Python/Dea.eps')   # save the figure to file
        #plt.close(fig)
        # ***********************************************************************************
    # Select train and test periods
    # *******

    sw_periods = '1.2.60'   # !!
    # '0.1': train over the whole data
    # '1.01': train period around the peak chosen "by hand" for BEL
    # '1.02': a few train periods around the peak chosen "by hand" for BEL
    # '1.1.60': put train period around peak WITH data leakage - do not use
    # '1.2.60': train period around the peak, selected automatically without data leakage
    # '2.1': train period around the end
    # '3.1': sliding train window, test until end
    # "3.1.60': sliding train window, test duration 60

    if sw_periods == '0.1':
        # One large train:  *keep*-with c_E=c_L=0, only show_H
        train_t_start_vals = np.array([1]);
        train_t_end_vals = (len(total_in)-0) * np.ones(np.shape(train_t_start_vals),dtype=int);
        test_t_end_vals = (len(total_in)-0) * np.ones(np.shape(train_t_start_vals),dtype=int);

    if sw_periods == '1.01':
        # This one gives a MAPE_test of 7.9% for Belgium
        train_t_start_vals = np.array([18])
        train_t_end_vals = train_t_start_vals + 14;  # + 14 for 14 days in train period
        test_t_end_vals = (len(total_in)-0) * np.ones(np.shape(train_t_start_vals),dtype=int); 

    if sw_periods == '1.02':
        # A few small trains:  *keep*-only show_H show_S
        #train_t_start_vals = np.array([10,12,14,16,18,20,22])
        train_t_start_vals = np.arange(16,22,2) #for Belgium, French
        #train_t_start_vals = np.arange(8,22,4) #for UK and Luxembourg because the peak rapidely obtained
        train_t_end_vals = (32) * np.ones(np.shape(train_t_start_vals),dtype=int)
        #train_t_end_vals = train_t_start_vals + 14
        test_t_end_vals = (len(total_in)-0) * np.ones(np.shape(train_t_start_vals),dtype=int)
    
    if sw_periods == '1.03':
            # A few small trains:  *keep*-only show_H show_S
            train_t_start_vals = np.arange(1,15,7) #for Belgium 
            train_t_end_vals = (15) * np.ones(np.shape(train_t_start_vals),dtype=int)
            #train_t_end_vals = train_t_start_vals + 14
            test_t_end_vals = (len(total_in)-0) * np.ones(np.shape(train_t_start_vals),dtype=int)

    if sw_periods == '1.1.60':  # put train period around peak WITH data leakage
        N = 7  # The window length of moving average will be 2N+1.
        total_in_MA = total_in * np.nan  # MA: moving average
        for t in range(0,len(total_in)):
            total_in_MA[t] = np.sum(total_in[max(0,t-N):min(t+N+1,len(total_in))]) / (min(t+N+1,len(total_in)) - max(0,t-N))
        t_max = np.argmax(total_in_MA)  # t_max is the position of the max of the MA
        train_t_start_vals = np.array([t_max-7])
        train_t_end_vals = train_t_start_vals + 15  # The train period is an interval of 15 days centered at the peak of total_in_MA
        test_t_end_vals = train_t_end_vals + 60
        # WARNING: In keeping with Python conventions, _end variables give the integer *before which* we stop.

        # if show_figures:   # plot total_in and total_in_MA
        #     plt.figure(figsize=(10,8))
        #     plt.plot(dates, total_in, "-", color='gray', label="Total_in")
        #     plt.plot(dates, total_in_MA, 'm--', label="total_in_MA")
        #     plt.xlabel("Dates")
        #     #plt.ylabel("Values")
        #     plt.legend()
        #     plt.show(block=False)

    if sw_periods == '1.2.60':  # put train period around peak while avoiding data leakage
        # *keep*
        N = 7  # The window length of moving average will be 2N+1.
        total_in_MA = total_in * np.nan  # MA: moving average
        # Find the first time where the latest peak of total_in_MA is N days behind:
        t = -1; t_max = -1
        while t_max != t-N or total_in_MA[t_max] == 0 or t-2*N < 0:
            t = t + 1
            total_in_MA[t] = np.sum(total_in[max(0,t-N):min(t+N+1,len(total_in))]) / (min(t+N+1,len(total_in)) - max(0,t-N))
            t_max = np.argmax(total_in_MA[:t+1])
        train_t_start_vals = np.array([t-2*N])
        train_t_end_vals = np.array([t+N+1])  # The train period stops the day before train_t_end_vals. Observe that the data from train_t_end_vals onward has not been used.
        test_t_end_vals = train_t_end_vals + 60       
    # # One small train period around peak:
    # train_t_start_vals = np.array([18]);    # For Belgium, dates[17] is 2020-04-01
    # train_t_end_vals = train_t_start_vals + 14;  # + 14 for 14 days in train period
    # test_t_end_vals = (len(total_in)-0) * np.ones(np.shape(train_t_start_vals),dtype=int); 

    # # Two small train periods:
    # train_t_start_vals = np.array([10,60]);    # For Belgium, dates[17] is 2020-04-01
    # train_t_end_vals = train_t_start_vals + 14;  # + 14 for 14 days in train period
    # test_t_end_vals = (len(total_in)-0) * np.ones(np.shape(train_t_start_vals),dtype=int);

    # # Sliding train and test windows:
    # train_t_start_vals = np.arange(1,len(total_in)-28,7)  
    # train_t_end_vals = train_t_start_vals + 14
    # test_t_end_vals = train_t_end_vals + 14 

    if sw_periods == '2.1':
        # Train period around the end.
        train_t_start_vals = np.array([1+10*7]);
        #train_t_start_vals = np.arange(1+11*7,len(total_in)-28,7) 
        train_t_end_vals = train_t_start_vals + 14
        test_t_end_vals = (len(total_in)-0) * np.ones(np.shape(train_t_start_vals),dtype=int);

    if sw_periods == '3.1':
        # Sliding train window: *keep* - also for FRA
        train_t_start_vals = np.arange(1,len(total_in)-28,7) 
        train_t_end_vals = train_t_start_vals + 14
        test_t_end_vals = len(total_in) * np.ones(np.shape(train_t_start_vals),dtype=int)

    if sw_periods == '3.1.60':
        # Sliding train window:
        train_t_start_vals = np.arange(1,len(total_in)-14-60,7) 
        train_t_end_vals = train_t_start_vals + 14
        test_t_end_vals = train_t_end_vals + 60

    # ***********************************************************************************
    # Preparation
    # *******
    nb_periods = len(train_t_start_vals)  # number of periods, i.e., number of test-train experiments on the same data    
    # Make sure that times are integers:
    train_t_start_vals = train_t_start_vals.astype(int)
    train_t_end_vals = train_t_end_vals.astype(int)
    test_t_end_vals = test_t_end_vals.astype(int)

    # Restrict test_t_end_vals from above by len(total_in):
    test_t_end_vals = np.minimum(test_t_end_vals,len(total_in))
    
    # Weights of the terms of the cost function:
    c_H, c_E, c_L= 1, 1, 1 # !! Default: c_H = 1; c_E = 1; c_L = 1 (it gives a good MAPE_test)
    c_HEL = [c_H,c_E,c_L]
    
    #***********************************************************************************
    # Define gamma estimation function
    # *******
    # Model for gamma: new_out = gamma * total_in
    def estimate_gamma(tspan_train,data_totinout_train):
        train_t_start = tspan_train[0]
        train_t_end = tspan_train[1]
        total_in_train = data_totinout_train[:,0]
        new_out_train = data_totinout_train[:,2]
        # Estimator by ratio of means:
        #gamma_hat_RM = np.sum(new_out_train[0:train_t_end])/np.sum(total_in_train[0:train_t_end]) # This version uses all the non-test data.
        gamma_hat_RM = np.sum(new_out_train[train_t_start:train_t_end])/np.sum(total_in_train[train_t_start:train_t_end])  # This version uses only the "train" period.
        # Estimator by least squares:
        #gamma_hat_LS = total_in_train[0:train_t_end]\new_out_train[0:train_t_end]
        gamma_hat_LS = np.linalg.lstsq(np.c_[total_in_train[0:train_t_end]],new_out_train[0:train_t_end], rcond=None)[0]
        # Estimator by ratio of means on all data (test and train):  not legitimate
        #gamma_hat_all_RM = sum(new_out_train)/sum(total_in_train);
        # I observe that the RM and LS estimates are quite close. Let's keep:
        gamma = gamma_hat_RM
        #gamma = gamma_hat_all_RM;  % not legitimate
        return gamma

    # Model for mu: death = mu * total_in
    def estimate_mu(tspan_train,data_totinout_train):
        train_t_start = tspan_train[0]
        train_t_end = tspan_train[1]
        new_out_train = data_totinout_train[:,2]
        death_train = data_totinout_train[:,3]

        # Estimator by ratio of means:
        #gamma_hat_RM = np.sum(new_out_train[0:train_t_end])/np.sum(total_in_train[0:train_t_end]) # This version uses all the non-test data.
        mu_hat_RM = np.sum(death_train[train_t_start:train_t_end])/np.sum(new_out_train[train_t_start:train_t_end])  # This version uses only the "train" period.
       
        # Estimator by least squares:
        #gamma_hat_LS = total_in_train[0:train_t_end]\new_out_train[0:train_t_end]
        mu_hat_LS = np.linalg.lstsq(np.c_[new_out_train[0:train_t_end]],death_train[0:train_t_end], rcond=None)[0]
        # Estimator by ratio of means on all data (test and train):  not legitimate
        mu = mu_hat_RM
        #gamma = gamma_hat_all_RM;  % not legitimate
        return mu   
    #***********************************************************************************
    # Define function for successive (instead of joint) estimation of beta_bar and S_bar_init
    # *******
    def estimate_successive_betabar_Sbarinit(tspan_train,data_totinout_train):
        
        train_t_start = tspan_train[0]
        train_t_end = tspan_train[1]
        total_in_train = data_totinout_train[:,0]
        new_in_train = data_totinout_train[:,1]

        # Estimator by ratio of means:
        beta_bar_hat_RM = (total_in_train[train_t_end-1]-total_in_train[train_t_start]) / np.sum(total_in_train[train_t_start:train_t_end-1]**2) - (new_in_train[train_t_end-1]-new_in_train[train_t_start]) / np.sum(total_in_train[train_t_start:train_t_end-1]*new_in_train[train_t_start:train_t_end-1])  # This is based on equation (13) of https://arxiv.org/abs/2007.10492.

        beta_bar = beta_bar_hat_RM
        S_bar_init = new_in_train[train_t_start] / (beta_bar * total_in_train[train_t_start])

        # When the estimation is done in the decreasing phase, beta_bar_hat_RM can be negative. See SHR_18PA.py_save07 for an example. We remedy it as follows.
        if beta_bar < 0:
            beta_bar = -beta_bar
            S_bar_init = -S_bar_init 

        return beta_bar, S_bar_init
    # ***********************************************************************************
    # Define several functions: the SH simulation function; the general cost function on which the various parameter estimations will be based; a function that returns statistics; a function that draw plots of simulation results
    # *******
    # Define the simulation function of the SH model:
    def simu(beta_bar,gamma,S_bar_init,H_init,tspan):
        simu_t_start = tspan[0]
        simu_t_end = tspan[1]  # The time before which we stop, i.e., the last returned values are at t = simu_t_end - 1.
        S_bar = np.full(simu_t_end, np.nan)  # set storage
        H = np.full(simu_t_end, np.nan)  # set storage
        E = np.full(simu_t_end, np.nan)  # set storage
        L = np.full(simu_t_end, np.nan)  # set storage
        #D = np.full(simu_t_end, np.nan)  # set storage
       # D_by_day = np.full(simu_t_end, np.nan)  # set storage
        S_bar[simu_t_start] = S_bar_init
        H[simu_t_start] = H_init
        E[simu_t_start] = beta_bar * S_bar[simu_t_start] * H[simu_t_start]
        L[simu_t_start] = gamma * H[simu_t_start]
        #D[simu_t_start] = D_init
        #D_by_day[simu_t_start]=mu * H[simu_t_start]
        for t in np.arange(simu_t_start,simu_t_end-1):
            S_bar[t+1] = S_bar[t] - beta_bar * S_bar[t] * H[t]
            H[t+1] = H[t] + beta_bar * S_bar[t] * H[t] - gamma* H[t]
            E[t+1] = beta_bar * S_bar[t+1] * H[t+1]
            L[t+1] = gamma * H[t+1]
            #D[t+1] =D[t] + mu * H[t]
            #D_by_day[t+1]=mu * L[t+1] # Pour que ça soit un flux sortant de L
        return (S_bar,H,E,L)
    # end def simu

    # Define the loss function in terms of all the possible decision variables, i.e., beta_bar,gamma,S_bar_init,H_init :
    def phi_basic(beta_bar,gamma,S_bar_init,H_init,tspan_train,data_totinout_train,c_HEL):
        # Extract variables from input:
        c_H, c_E, c_L = c_HEL  #coefficients of the terms of the cost function. Default: c_H = 1; c_E = 1; c_L = 1 (it gives a good MAPE_test)
        train_t_start, train_t_end = tspan_train
        _, H, E, L = simu(beta_bar,gamma,S_bar_init,H_init,tspan=tspan_train)  # "_" because S_bar is not involved in the cost
        # Compute the cost (discrepancy between observed and simulated):
        cost = c_H * (np.linalg.norm(H[train_t_start:train_t_end]-data_totinout_train[train_t_start:train_t_end,0]))**2 + c_E * (np.linalg.norm(E[train_t_start:train_t_end]-data_totinout_train[train_t_start:train_t_end,1]))**2 + c_L * (np.linalg.norm(L[train_t_start+1:train_t_end]-data_totinout_train[train_t_start+1:train_t_end,2]))**2 
        return cost
    
    # Define function for plots:
    def make_plots(beta_bar,gamma,S_bar_init,H_init,mu,tspan_train,dates,data_totinout):
        S_bar, H, E, L = simu(beta_bar, gamma, S_bar_init, H_init, tspan=[tspan_train[0],len(total_in)])
        nb_subplots = show_H + show_S_bar + show_beta_bar + show_gamma + show_mu + show_D_by_day
        if nb_subplots == 6:
            nb_subplot_rows = 2
            nb_subplot_cols = 3
            plt.rc('xtick', labelsize='x-small')
            plt.rc('ytick', labelsize='x-small')
        else:
            nb_subplot_rows = 1
            nb_subplot_cols = nb_subplots 
        cnt_subplot = 0
        nb_xticks = 4
        dates_ticks = [None] * nb_xticks
        dates_ticks_ind = np.linspace(0,len(total_in)-1,nb_xticks,dtype=int)
        for i in range(0,nb_xticks):
            dates_ticks[i] = dates[dates_ticks_ind[i]]

        if show_H:
            cnt_subplot = cnt_subplot + 1
            plt.subplot(nb_subplot_rows,nb_subplot_cols,cnt_subplot)

            if cnt_period == 0:   # assign plot labels
                plt.plot(dates,total_in, "-", color='gray', label="Total_in", linewidth=1)
                plt.plot(dates[train_t_start:train_t_end],H[train_t_start:train_t_end],'b--', label="H_train")
                plt.plot(dates[train_t_end-1:test_t_end],H[train_t_end-1:test_t_end],'r-.', label="H_pred")
                plt.legend()
            else:
                plt.plot(dates[train_t_start:train_t_end],H[train_t_start:train_t_end],'b--')
                plt.plot(dates[train_t_end-1:test_t_end],H[train_t_end-1:test_t_end],'r-.')
            plt.xticks(dates_ticks)
            plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
            if cnt_period == nb_periods-1:
                plt.ylim(bottom=0)

        if show_S_bar:
            cnt_subplot = cnt_subplot + 1
            plt.subplot(nb_subplot_rows,nb_subplot_cols,cnt_subplot)
            if cnt_period == 0:   # assign plot labels
                plt.plot(dates[0:train_t_end],S_bar[0:train_t_end],'b--', label="S_bar_train")
                plt.plot(dates[train_t_end-1:test_t_end],S_bar[train_t_end-1:test_t_end],'r-.', label="S_bar_pred")

                plt.legend()
            else:
                plt.plot(dates[train_t_start:train_t_end],S_bar[train_t_start:train_t_end],'b--')
                plt.plot(dates[train_t_end-1:test_t_end],S_bar[train_t_end-1:test_t_end],'r-.')
            plt.xticks(dates_ticks)
            plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
            if cnt_period == nb_periods-1:
                plt.ylim(bottom=0)

        if show_beta_bar:
            cnt_subplot = cnt_subplot + 1
            plt.subplot(nb_subplot_rows,nb_subplot_cols,cnt_subplot)
            if cnt_period == 0:   # assign plot labels
                plt.plot(dates[train_t_start:train_t_end],beta_bar*np.ones(train_t_end-train_t_start),'b--', label="beta_bar_train")
                plt.plot(dates[train_t_end-1:test_t_end],beta_bar*np.ones(test_t_end-train_t_end+1),'r-.', label="beta_bar_pred")
                plt.legend()
            else:
                plt.plot(dates[train_t_start:train_t_end],beta_bar*np.ones(train_t_end-train_t_start),'b--')
                plt.plot(dates[train_t_end-1:test_t_end],beta_bar*np.ones(test_t_end-train_t_end+1),'r-.')
            plt.xticks(dates_ticks)
            plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
            if cnt_period == nb_periods-1:
                plt.ylim(bottom=0)

        if show_gamma:
            cnt_subplot = cnt_subplot + 1
            plt.subplot(nb_subplot_rows,nb_subplot_cols,cnt_subplot)
            if cnt_period == 0:   # assign plot labels
                plt.plot(dates[train_t_start:train_t_end],gamma*np.ones(train_t_end-train_t_start),'b--', label="gamma_train")
                plt.plot(dates[train_t_end-1:test_t_end],gamma*np.ones(test_t_end-train_t_end+1),'r-.', label="gamma_pred")
                plt.legend()
            else:
                plt.plot(dates[train_t_start:train_t_end],gamma*np.ones(train_t_end-train_t_start),'b--')
                plt.plot(dates[train_t_end-1:test_t_end],gamma*np.ones(test_t_end-train_t_end+1),'r-.')
            plt.xticks(dates_ticks)
            plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
            if cnt_period == nb_periods-1:
                plt.ylim(bottom=0)
                
        if show_mu:
            cnt_subplot = cnt_subplot + 1
            plt.subplot(nb_subplot_rows,nb_subplot_cols,cnt_subplot)
            if cnt_period == 0:   # assign plot labels
                plt.plot(dates[train_t_start:train_t_end],mu*np.ones(train_t_end-train_t_start),'b--', label="mu_train")
                plt.plot(dates[train_t_end-1:test_t_end],mu*np.ones(test_t_end-train_t_end+1),'r-.', label="mu_pred")
                plt.legend()
            else:
                plt.plot(dates[train_t_start:train_t_end],mu*np.ones(train_t_end-train_t_start),'b--')
                plt.plot(dates[train_t_end-1:test_t_end],mu*np.ones(test_t_end-train_t_end+1),'r-.')
            plt.xticks(dates_ticks)
            plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
            if cnt_period == nb_periods-1:
                plt.ylim(bottom=0)
                
        if show_D_by_day:
            cnt_subplot = cnt_subplot + 1
            plt.subplot(nb_subplot_rows,nb_subplot_cols,cnt_subplot)
            D_by_day=mu*L
            if cnt_period == 0:   # assign plot labels    
                plt.plot(dates,death, "-", color='black', label="Deaths", linewidth=1)
                plt.plot(dates[train_t_start:train_t_end],D_by_day[train_t_start:train_t_end],'b--', label="D_train")
                plt.plot(dates[train_t_end-1:test_t_end],D_by_day[train_t_end-1:test_t_end],'r-.', label="D_pred")
                plt.legend()
            else:
                plt.plot(dates[train_t_start:train_t_end],D_by_day[train_t_start:train_t_end],'b--')
                plt.plot(dates[train_t_end-1:test_t_end],D_by_day[train_t_end-1:test_t_end],'r-.')
            plt.xticks(dates_ticks)
            plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
            if cnt_period == nb_periods-1:
                plt.ylim(bottom=0)
        return
    # end def make_plots
    
    def phi(x,gamma,H_init,tspan_train,data_totinout_train,c_HEL):
        return phi_basic(x[0],gamma,x[1],H_init,tspan_train,data_totinout_train,c_HEL)

    # Extract train variables in order to do a first plot of the cost function:
    train_t_start = train_t_start_vals[0]
    train_t_end = train_t_end_vals[0]
    test_t_end = test_t_end_vals[0]
    tspan_train = [train_t_start,train_t_end]
    data_totinout_train = copy.deepcopy(data_totinout)  # in order to be able to "hide" entries in data_totinout_train without changing data_totinout
    data_totinout_train[train_t_end:,:] = np.nan  # Beware that data_totinout_train has to be floats.

    # Define anonymous function for used in optimization solver:
    H_init = data_totinout_train[tspan_train[0],0]
    #D_init = data_totinout_train[tspan_train[0],3]
    gamma = estimate_gamma(tspan_train,data_totinout_train)  # estimate gamma
    #gamma = estimate_gammaKM(tspan_train,data_totinout_train)  # estimate gamma
    mu= estimate_mu(tspan_train,data_totinout_train)  # estimate mu
    #mu= estimate_mu_obs(tspan_train,data_totinout_train)  # estimate mu_obs
    
    
    # Define function for the estimation of beta_bar and S_bar_init:
    def estimate_betabar_Sbarinit(H_init,tspan_train,data_totinout_train,c_HEL):  # estimation method with beta_bar and S_bar_init as optimization variables
        # Estimate gamma:
        gamma = estimate_gamma(tspan_train,data_totinout_train)
        mu = estimate_mu(tspan_train,data_totinout_train)
        fun = lambda x:phi(x,gamma,H_init,tspan_train,data_totinout_train,c_HEL)  # function phi is defined above
        beta_bar_guess, S_bar_init_guess = estimate_successive_betabar_Sbarinit(tspan_train,data_totinout_train)
        x_guess = [beta_bar_guess, S_bar_init_guess]
        #x_guess = [1e-5,1e4]  # hard-coded guess. Sugg: [1e-5,1e4] 
        x_opt = optimize.fmin(fun,x_guess)  # call the optimization solver
        beta_bar_opt = x_opt[0]
        S_bar_init_opt = x_opt[1]
        fun_opt = fun([beta_bar_opt,S_bar_init_opt])  # value of the minimum, useful for plot
        return (beta_bar_opt, S_bar_init_opt, gamma, mu,fun_opt, fun, x_guess)

    if show_figures:
        plt.figure(figsize=(10,8))
    # Storage for statistics as a dictionary of numpy arrays:
    # If we are dealing with the first district, then we have to create stats_all 
    if cnt_district == 0:   
        stats_all = {}   # Create dict stats_all. Using make_stats(s), it will become a dictionary of numpy arrays where we will record statistics for the various districts and periods.
    # [[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[
    # Start loop on train periods:
    for cnt_period in range(0, nb_periods):
        # Extract train variables for period cnt_period:
        train_t_start = train_t_start_vals[cnt_period]
        train_t_end = train_t_end_vals[cnt_period]
        test_t_end = test_t_end_vals[cnt_period]
        tspan_train = [train_t_start,train_t_end]
        # The test data is defined to be all the data that occurs from train_t_end.
        # Replace test data by NaN in *_train variables.
        data_totinout_train = copy.deepcopy(data_totinout)  # in order to be able to "hide" entries in data_totinout_train without changing data_totinout
        data_totinout_train[train_t_end:,:] = np.nan  # Beware that data_totinout_train has to be floats.
        # ! Make sure to use only these *_train variables in the train phase.
        H_init = data_totinout_train[tspan_train[0],0]

        # Estimate beta_bar and S_bar_init by optimizing the cost function:
        beta_bar_opt, S_bar_init_opt, gamma, mu, fun_opt, fun, x_guess = estimate_betabar_Sbarinit(H_init,tspan_train,data_totinout_train,c_HEL)
        # Plot true and simulated H, and simulated S_bar:
        if show_figures:
             make_plots(beta_bar_opt,gamma,S_bar_init_opt,H_init,mu,tspan_train,dates,data_totinout)  
        #end if cnt_period      
    # end loop on train periods
    # ]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]

    if show_figures:
        #plt.title("Optimized wrt beta_bar and S_bar_init")
        plt.show(block=False)  
    
    #test plots
    if show_figures:
        fig=plt.figure(figsize=(16,8))
    for cnt_period in range(0, nb_periods):
        # Extract train variables for period cnt_period:
        train_t_start = train_t_start_vals[cnt_period]
        train_t_end = train_t_end_vals[cnt_period]
        test_t_end = test_t_end_vals[cnt_period]
        tspan_train = [train_t_start,train_t_end]
        # The test data is defined to be all the data that occurs from train_t_end.
        # Replace test data by NaN in *_train variables.
        data_totinout_train = copy.deepcopy(data_totinout)  # in order to be able to "hide" entries in data_totinout_train without changing data_totinout
        data_totinout_train[train_t_end:,:] = np.nan  # Beware that data_totinout_train has to be floats.
        # ! Make sure to use only these *_train variables in the train phase.
        H_init = data_totinout_train[tspan_train[0],0]
        # Estimate beta_bar and S_bar_init by optimizing the cost function:
        beta_bar_opt, S_bar_init_opt, gamma, mu, fun_opt, fun, x_guess = estimate_betabar_Sbarinit(H_init,tspan_train,data_totinout_train,c_HEL)
        # Plot true and simulated H, and simulated S_bar:
        S_bar, H, E, L = simu(beta_bar_opt, gamma, S_bar_init_opt, H_init, tspan=[tspan_train[0],len(total_in)])
        nb_subplots = show_H + show_S_bar 
        if nb_subplots == 1:
            nb_subplot_rows = 1
            nb_subplot_cols = 1
            plt.rc('xtick', labelsize='x-small')
            plt.rc('ytick', labelsize='x-small')
        else:
            nb_subplot_rows = 1
            nb_subplot_cols = nb_subplots 
        cnt_subplot = 0
        nb_xticks = 4
        dates_ticks = [None] * nb_xticks
        dates_ticks_ind = np.linspace(0,len(total_in)-1,nb_xticks,dtype=int)
        for i in range(0,nb_xticks):
            dates_ticks[i] = dates[dates_ticks_ind[i]]  
        if show_H:
            cnt_subplot = cnt_subplot + 1
            plt.subplot(nb_subplot_rows,nb_subplot_cols,cnt_subplot)
            if cnt_period == 0:   # assign plot labels
                plt.plot(dates,total_in, "-", color='gray', label="Total_in", linewidth=1)
                plt.plot(dates[train_t_start:train_t_end],H[train_t_start:train_t_end],'b--', label="H_train")
                plt.plot(dates[train_t_end-1:test_t_end],H[train_t_end-1:test_t_end],'r-.', label="H_pred")
                plt.xlabel("Dates", fontsize=15)
                plt.ylabel("Hospitalization cases", fontsize=15)
                plt.legend(fontsize=15)
                plt.tick_params(labelsize=14)
            else:
                plt.plot(dates[train_t_start:train_t_end],H[train_t_start:train_t_end],'b--')
                plt.plot(dates[train_t_end-1:test_t_end],H[train_t_end-1:test_t_end],'r-.')
            plt.xticks(dates_ticks)
           # plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0), fontsize=12)
            if cnt_period == nb_periods-1:
                plt.ylim(bottom=0)
    #fig.savefig('C:/Users/odiao/Desktop/Presentation_Latex/Benelux_meeting/Figures/SHR_DEATH_OD3_H_and_S_bar.pdf')   # save the figure to file
    #plt.close(fig)
        if show_S_bar:
            cnt_subplot = cnt_subplot + 1
            plt.subplot(nb_subplot_rows,nb_subplot_cols,cnt_subplot)
            if cnt_period == 0:   # assign plot labels
                plt.plot(dates[0:train_t_end],S_bar[0:train_t_end],'b--', label="S_bar_train")
                plt.plot(dates[train_t_end-1:test_t_end],S_bar[train_t_end-1:test_t_end],'r-.', label="S_bar_pred")
                
                plt.xlabel("Dates", fontsize=15)
                plt.ylabel("S_bar values", fontsize=15)
                plt.legend(fontsize=15)
                plt.tick_params(labelsize=14)
            else:
                plt.plot(dates[train_t_start:train_t_end],S_bar[train_t_start:train_t_end],'b--')
                plt.plot(dates[train_t_end-1:test_t_end],S_bar[train_t_end-1:test_t_end],'r-.')
            plt.xticks(dates_ticks)
            #plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0), fontsize=12)
            if cnt_period == nb_periods-1:
                plt.ylim(bottom=0)
   
    #fig.savefig('C:/Users/odiao/Desktop/Presentation_Latex/Benelux_meeting/Figures/SHR_DEATH_OD3_H_and_S_bar.pdf')   # save the figure to file
    #plt.close(fig)
#***************************************************************************************
#***************************************************************
#Least squares to estiate gamma, N_bar=S_bar_init+Hinit and beta_bar
    #Daily discharged as function cumulative discharged
#***************************************************************
    from scipy.optimize import least_squares
    fig=plt.figure(figsize=(12,6))
    #gamma = 0.0698
    S_bar_init_opt=17800#15344
    H_init=370.0
    beta_bar_opt=1.7024780797870242e-05 
    N_bar=S_bar_init_opt+H_init
    R_bar=np.cumsum(new_out)
    R_day=gamma*N_bar*(1-np.exp(-beta_bar_opt*R_bar/gamma))-gamma*R_bar
    
    plt.plot(R_bar, new_out, 'o', markersize=4, label='data')
    plt.plot(R_bar, R_day, label='fitted model')
    plt.xlabel("Cumulative number of discharged")
    plt.ylabel("Daily discharged")
    plt.legend()
    plt.show()
    #fig.savefig('C:/Users/odiao/Desktop/Redaction_darticles_Latex/SHR_PA/Figures/R_bar.pdf')   # save the figure to file
    #plt.close(fig)
    #************Optimization
    fig=plt.figure(figsize=(12,6))
    def modell(x, u):
        return x[0]*x[1]*(1-np.exp(-x[2]*u/x[0]))-x[0]*u
    def funn(x, u, y):
        return modell(x, u) - y  
    y=new_out
    u=np.cumsum(new_out)  
    x0=[gamma, N_bar,beta_bar_opt]
    ress = least_squares(funn,x0, args=(u, y), verbose=1)
    u_test = u
    y_test = modell(ress.x, u_test)
    
    plt.plot(u, y, 'o', markersize=4, label='Real data')
    plt.plot(u_test, y_test, label='Fitted model')
    plt.xlabel("Cumulative number of discharged", fontsize=20)
    plt.ylabel("Daily discharged", fontsize=20)
    plt.legend(fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.show()
    param=[ress.x[0], ress.x[1], ress.x[2]]
    #fig.savefig('C:/Users/odiao/Desktop/Model Covid19/programme_python/SHR_PA/Code_Python/R_bar_opt.eps')   # save the figure to file
    #plt.close(fig)  
    
    if show_figures:
        fig=plt.figure(figsize=(10,8))
    for cnt_period in range(0, nb_periods):
        # Extract train variables for period cnt_period:
        train_t_start = train_t_start_vals[cnt_period]
        train_t_end = train_t_end_vals[cnt_period]
        test_t_end = test_t_end_vals[cnt_period]
        tspan_train = [train_t_start,train_t_end]
        # The test data is defined to be all the data that occurs from train_t_end.
        # Replace test data by NaN in *_train variables.
        data_totinout_train = copy.deepcopy(data_totinout)  # in order to be able to "hide" entries in data_totinout_train without changing data_totinout
        data_totinout_train[train_t_end:,:] = np.nan  # Beware that data_totinout_train has to be floats.
        # ! Make sure to use only these *_train variables in the train phase.
        H_init = data_totinout_train[tspan_train[0],0]
        #New_out_init = data_totinout_train[tspan_train[0],2]
        gamma_D_bar=ress.x[0]
        S_bar_init_opt_D_bar = ress.x[1] - H_init  #- New_out_init
        beta_bar_opt_D_bar = ress.x[2]
        # Plot true and simulated H, and simulated S_bar:
        S_bar, H, E, L = simu(beta_bar_opt, gamma, S_bar_init_opt, H_init, tspan=[tspan_train[0],len(total_in)])
        nb_subplots = show_H + show_S_bar 
        if nb_subplots == 2:
            nb_subplot_rows = 1
            nb_subplot_cols = 1
            plt.rc('xtick', labelsize='x-small')
            plt.rc('ytick', labelsize='x-small')
        else:
            nb_subplot_rows = 1
            nb_subplot_cols = nb_subplots 
        cnt_subplot = 0
        nb_xticks = 4
        dates_ticks = [None] * nb_xticks
        dates_ticks_ind = np.linspace(0,len(total_in)-1,nb_xticks,dtype=int)
        for i in range(0,nb_xticks):
            dates_ticks[i] = dates[dates_ticks_ind[i]]
    
        if show_L:
            cnt_subplot = cnt_subplot + 1
            plt.subplot(nb_subplot_rows,nb_subplot_cols,cnt_subplot)
            if cnt_period == 0:   # assign plot labels    
                plt.plot(dates,new_out, "-", color='black', label="Discharged", linewidth=1)
                plt.plot(dates[train_t_start:train_t_end],L[train_t_start:train_t_end],'b--', label="Dis_train")
                plt.plot(dates[train_t_end-1:test_t_end],L[train_t_end-1:test_t_end],'r-.', label="Dis_pred")
                plt.legend(fontsize=20)
            else:
                plt.plot(dates[train_t_start:train_t_end],L[train_t_start:train_t_end],'b--')
                plt.plot(dates[train_t_end-1:test_t_end],L[train_t_end-1:test_t_end],'r-.')
            plt.xticks(dates_ticks, fontsize=15)
            plt.yticks(fontsize=15)
            #plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
            if cnt_period == nb_periods-1:
                plt.ylim(bottom=0)
    #fig.savefig('C:/Users/odiao/Desktop/Model Covid19/programme_python/SHR_PA/Code_Python/Rpred_bar_opt.eps')   # save the figure to file
    #plt.close(fig)
    
    
#***************************************************************
#Least squares to estiate gamma, N_bar=S_bar_init+Hinit and beta_bar
#***************************************************************
    #from scipy.optimize import least_squares
  #np.sum(new_out_raw+death)/np.sum(total_in)
    #gamma = gamma_R_bar#np.sum(new_out)/np.sum(total_in)
    CFR = np.sum(death)/(np.sum(death)+np.sum(new_out))
    p = CFR/(1-CFR)
    S_bar_init_opt=17800
    H_init=370.0
    beta_bar_opt= 1.5e-05 
    N_bar=S_bar_init_opt+H_init
    D_bar=np.cumsum(death)
    D_day=p*gamma*N_bar*(1-np.exp(-(beta_bar_opt*D_bar)/(p*gamma)))-gamma*D_bar
    fig=plt.figure(figsize=(12,6))
    plt.plot(D_bar, death, 'o', markersize=4, label='data')
    plt.plot(D_bar, D_day, label='fitted model')
    plt.xlabel("Cumulative number of fatalities", fontsize=16)
    plt.ylabel("Daily fatalities", fontsize=16)
    plt.legend(fontsize=16)
    plt.tick_params(labelsize=14)
    plt.show()
    #fig.savefig('C:/Users/odiao/Desktop/Redaction_darticles_Latex/SHR_PA/Figures/fd.pdf')   # save the figure to file
    #plt.close(fig)
    #************Optimization
    def modell(x, u):
        return x[0]*x[1]*x[2]*(1-np.exp(-(x[3]*u)/(x[0]*x[1])))-x[1]*u
    def funn(x, u, y):
        return modell(x, u) - y  
    y=death
    u=np.cumsum(death)  
    x0=[p,gamma, N_bar,beta_bar_opt]
    res = least_squares(funn,x0, args=(u, y), verbose=1)
    u_test = u
    y_test = modell(res.x, u_test)
    fig=plt.figure(figsize=(12,6))
    plt.plot(u, y, 'o', markersize=4, label='Real data')
    plt.plot(u_test, y_test, label='Fitted model')
    plt.xlabel("Cumulative number of fatalities", fontsize=20)
    plt.ylabel("Daily fatalities", fontsize=20)
    plt.legend(fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.show()
    #fig.savefig('C:/Users/odiao/Desktop/Model Covid19/programme_python/SHR_PA/Code_Python/fd_opt.eps')   # save the figure to file
    #plt.close(fig)
 #****************************************************************   
     #tester les differentes valeurs   
        #test plots show_H and show_S_bar
    if show_figures:
        fig=plt.figure(figsize=(10,8))
    for cnt_period in range(0, nb_periods):
        # Extract train variables for period cnt_period:
        train_t_start = train_t_start_vals[cnt_period]
        train_t_end = train_t_end_vals[cnt_period]
        test_t_end = test_t_end_vals[cnt_period]
        tspan_train = [train_t_start,train_t_end]
        # The test data is defined to be all the data that occurs from train_t_end.
        # Replace test data by NaN in *_train variables.
        data_totinout_train = copy.deepcopy(data_totinout)  # in order to be able to "hide" entries in data_totinout_train without changing data_totinout
        data_totinout_train[train_t_end:,:] = np.nan  # Beware that data_totinout_train has to be floats.
        # ! Make sure to use only these *_train variables in the train phase.
        H_init = data_totinout_train[tspan_train[0],0]
        #New_out_init = data_totinout_train[tspan_train[0],2]
        p_D_bar=res.x[0]
        gamma_D_bar=res.x[1]
        S_bar_init_opt_D_bar = res.x[2] - H_init  #- New_out_init
        beta_bar_opt_D_bar = res.x[3]
        # Plot true and simulated H, and simulated S_bar:
        S_bar, H, E, L = simu(beta_bar_opt, gamma, S_bar_init_opt, H_init, tspan=[tspan_train[0],len(total_in)])
        nb_subplots = show_H + show_S_bar 
        if nb_subplots == 2:
            nb_subplot_rows = 1
            nb_subplot_cols = 2
            plt.rc('xtick', labelsize='x-small')
            plt.rc('ytick', labelsize='x-small')
        else:
            nb_subplot_rows = 1
            nb_subplot_cols = nb_subplots 
        cnt_subplot = 0
        nb_xticks = 4
        dates_ticks = [None] * nb_xticks
        dates_ticks_ind = np.linspace(0,len(total_in)-1,nb_xticks,dtype=int)
        for i in range(0,nb_xticks):
            dates_ticks[i] = dates[dates_ticks_ind[i]]  
        if show_H:
            cnt_subplot = cnt_subplot + 1
            plt.subplot(nb_subplot_rows,nb_subplot_cols,cnt_subplot)
            if cnt_period == 0:   # assign plot labels
                plt.plot(dates,total_in, "-", color='gray', label="Total_in", linewidth=1)
                plt.plot(dates[train_t_start:train_t_end],H[train_t_start:train_t_end],'b--', label="H_train")
                plt.plot(dates[train_t_end-1:test_t_end],H[train_t_end-1:test_t_end],'r-.', label="H_pred")
                plt.xlabel("Dates", size=10)
                plt.ylabel("Hospitalization cases", size=10)
                plt.legend()
            else:
                plt.plot(dates[train_t_start:train_t_end],H[train_t_start:train_t_end],'b--')
                plt.plot(dates[train_t_end-1:test_t_end],H[train_t_end-1:test_t_end],'r-.')
            plt.xticks(dates_ticks)
            plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
            if cnt_period == nb_periods-1:
                plt.ylim(bottom=0)
    
        if show_S_bar:
            cnt_subplot = cnt_subplot + 1
            plt.subplot(nb_subplot_rows,nb_subplot_cols,cnt_subplot)
            if cnt_period == 0:   # assign plot labels
                plt.plot(dates[0:train_t_end],S_bar[0:train_t_end],'b--', label="S_bar_train")
                plt.plot(dates[train_t_end-1:test_t_end],S_bar[train_t_end-1:test_t_end],'r-.', label="S_bar_pred")
                plt.xlabel("Dates", size=10)
                plt.ylabel("S_bar values", size=10)
                plt.legend()
            else:
                plt.plot(dates[train_t_start:train_t_end],S_bar[train_t_start:train_t_end],'b--')
                plt.plot(dates[train_t_end-1:test_t_end],S_bar[train_t_end-1:test_t_end],'r-.')
            plt.xticks(dates_ticks)
            plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
            if cnt_period == nb_periods-1:
                plt.ylim(bottom=0)
   
    #fig.savefig('C:/Users/odiao/Desktop/Presentation_Latex/Benelux_meeting/Figures/SHR_DEATH_OD3_H_and_S_bar.pdf')   # save the figure to file
    #plt.close(fig)
          
    def make_plots_D_bar(beta_bar,gamma,S_bar_init,H_init,tspan_train,dates,data_totinout):
        S_bar, H, E, L = simu(beta_bar, gamma, S_bar_init, H_init, tspan=[tspan_train[0],len(total_in)])
        nb_subplots =  show_H + show_D_by_day
        if nb_subplots == 2:
            nb_subplot_rows = 1
            nb_subplot_cols = 2
            plt.rc('xtick', labelsize='x-small')
            plt.rc('ytick', labelsize='x-small')
        else:
            nb_subplot_rows = 1
            nb_subplot_cols = nb_subplots 
        cnt_subplot = 0
        nb_xticks = 4
        dates_ticks = [None] * nb_xticks
        dates_ticks_ind = np.linspace(0,len(total_in)-1,nb_xticks,dtype=int)
        for i in range(0,nb_xticks):
            dates_ticks[i] = dates[dates_ticks_ind[i]]

        if show_H:
            cnt_subplot = cnt_subplot + 1
            plt.subplot(nb_subplot_rows,nb_subplot_cols,cnt_subplot)
            if cnt_period == 0:   # assign plot labels    
                plt.plot(dates,total_in, "-", color='black', label="Hospitalized", linewidth=1)
                plt.plot(dates[train_t_start:train_t_end],H[train_t_start:train_t_end],'b--', label="Hos_train")
                plt.plot(dates[train_t_end-1:test_t_end],H[train_t_end-1:test_t_end],'r-.', label="Hos_pred")
                plt.xlabel("Dates", fontsize=15)
                plt.ylabel("Hospitalization cases", fontsize=15)
                plt.legend(fontsize=15)
                plt.tick_params(labelsize=14)
            else:
                plt.plot(dates[train_t_start:train_t_end],H[train_t_start:train_t_end],'b--')
                plt.plot(dates[train_t_end-1:test_t_end],H[train_t_end-1:test_t_end],'r-.')
            plt.xticks(dates_ticks)
            #plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
            if cnt_period == nb_periods-1:
                plt.ylim(bottom=0)             
        if show_D_by_day:
            D_by_day=p_D_bar*L
            cnt_subplot = cnt_subplot + 1
            plt.subplot(nb_subplot_rows,nb_subplot_cols,cnt_subplot)
            if cnt_period == 0:   # assign plot labels    
                plt.plot(dates,death, "-", color='black', label="Deaths", linewidth=1)
                plt.plot(dates[train_t_start:train_t_end],D_by_day[train_t_start:train_t_end],'b--', label="D_train")
                plt.plot(dates[train_t_end-1:test_t_end],D_by_day[train_t_end-1:test_t_end],'r-.', label="D_pred")
                plt.xlabel("Dates", fontsize=15)
                plt.ylabel("Daily COVID-19 deaths", fontsize=15)
                plt.legend(fontsize=15)
                plt.tick_params(labelsize=14)
            else:
                plt.plot(dates[train_t_start:train_t_end],D_by_day[train_t_start:train_t_end],'b--')
                plt.plot(dates[train_t_end-1:test_t_end],D_by_day[train_t_end-1:test_t_end],'r-.')
            plt.xticks(dates_ticks)
           # plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
            if cnt_period == nb_periods-1:
                plt.ylim(bottom=0)
        return  
    
    
    if show_figures:
       fig= plt.figure(figsize=(16,8))
    for cnt_period in range(0, nb_periods):
        # Extract train variables for period cnt_period:
        train_t_start = train_t_start_vals[cnt_period]
        train_t_end = train_t_end_vals[cnt_period]
        test_t_end = test_t_end_vals[cnt_period]
        tspan_train = [train_t_start,train_t_end]
        # The test data is defined to be all the data that occurs from train_t_end.
        # Replace test data by NaN in *_train variables.
        data_totinout_train = copy.deepcopy(data_totinout)  # in order to be able to "hide" entries in data_totinout_train without changing data_totinout
        data_totinout_train[train_t_end:,:] = np.nan  # Beware that data_totinout_train has to be floats.
        # ! Make sure to use only these *_train variables in the train phase.
        H_init = data_totinout_train[tspan_train[0],0]
        p_D_bar=res.x[0]
        gamma_D_bar=  res.x[1]
        S_bar_init_opt_D_bar = res.x[2] - H_init
        beta_bar_opt_D_bar = res.x[3]
        if show_figures:
             make_plots_D_bar(beta_bar_opt_D_bar,gamma_D_bar,S_bar_init_opt_D_bar,H_init,tspan_train,dates,data_totinout)  
    if show_figures:
        #plt.title("Optimized wrt beta_bar and S_bar_init")
        plt.show(block=False)   
   # fig.savefig('C:/Users/odiao/Desktop/Presentation_Latex/Benelux_meeting/Figures/SHR_DEATH_OD3_H_and_deaths_2021_05_14.pdf')   # save the figure to file
   # plt.close(fig)
    
#*****************************************************************************       
        # Define function for plots:
    def make_plots_D_bar(beta_bar,gamma,S_bar_init,H_init,tspan_train,dates,data_totinout):
        S_bar, H, E, L = simu(beta_bar, gamma, S_bar_init, H_init, tspan=[tspan_train[0],len(total_in)])
        nb_subplots =  show_L + show_D_by_day
        if nb_subplots == 2:
            nb_subplot_rows = 1
            nb_subplot_cols = 2
            plt.rc('xtick', labelsize='x-small')
            plt.rc('ytick', labelsize='x-small')
        else:
            nb_subplot_rows = 1
            nb_subplot_cols = nb_subplots 
        cnt_subplot = 0
        nb_xticks = 4
        dates_ticks = [None] * nb_xticks
        dates_ticks_ind = np.linspace(0,len(total_in)-1,nb_xticks,dtype=int)
        for i in range(0,nb_xticks):
            dates_ticks[i] = dates[dates_ticks_ind[i]]

        if show_L:
            cnt_subplot = cnt_subplot + 1
            plt.subplot(nb_subplot_rows,nb_subplot_cols,cnt_subplot)
            if cnt_period == 0:   # assign plot labels    
                plt.plot(dates,new_out, "-", color='black', label="Discharged", linewidth=1)
                plt.plot(dates[train_t_start:train_t_end],L[train_t_start:train_t_end],'b--', label="Dis_train")
                plt.plot(dates[train_t_end-1:test_t_end],L[train_t_end-1:test_t_end],'r-.', label="Dis_pred")
                plt.legend(fontsize=20)
            else:
                plt.plot(dates[train_t_start:train_t_end],L[train_t_start:train_t_end],'b--')
                plt.plot(dates[train_t_end-1:test_t_end],L[train_t_end-1:test_t_end],'r-.')
            plt.xticks(dates_ticks, fontsize=15)
            plt.yticks(fontsize=15)
            #plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
            if cnt_period == nb_periods-1:
                plt.ylim(bottom=0)
                
        if show_D_by_day:
            D_by_day=p_D_bar*L
            cnt_subplot = cnt_subplot + 1
            plt.subplot(nb_subplot_rows,nb_subplot_cols,cnt_subplot)
            if cnt_period == 0:   # assign plot labels    
                plt.plot(dates,death, "-", color='black', label="Deaths", linewidth=1)
                plt.plot(dates[train_t_start:train_t_end],D_by_day[train_t_start:train_t_end],'b--', label="D_train")
                plt.plot(dates[train_t_end-1:test_t_end],D_by_day[train_t_end-1:test_t_end],'r-.', label="D_pred")
                plt.legend(fontsize=20)
            else:
                plt.plot(dates[train_t_start:train_t_end],D_by_day[train_t_start:train_t_end],'b--')
                plt.plot(dates[train_t_end-1:test_t_end],D_by_day[train_t_end-1:test_t_end],'r-.')
            plt.xticks(dates_ticks, fontsize=15)
            plt.yticks(fontsize=15)
            #plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
            if cnt_period == nb_periods-1:
                plt.ylim(bottom=0)
        return
   
    if show_figures:
       fig= plt.figure(figsize=(16,8))
    for cnt_period in range(0, nb_periods):
        # Extract train variables for period cnt_period:
        train_t_start = train_t_start_vals[cnt_period]
        train_t_end = train_t_end_vals[cnt_period]
        test_t_end = test_t_end_vals[cnt_period]
        tspan_train = [train_t_start,train_t_end]
        # The test data is defined to be all the data that occurs from train_t_end.
        # Replace test data by NaN in *_train variables.
        data_totinout_train = copy.deepcopy(data_totinout)  # in order to be able to "hide" entries in data_totinout_train without changing data_totinout
        data_totinout_train[train_t_end:,:] = np.nan  # Beware that data_totinout_train has to be floats.
        # ! Make sure to use only these *_train variables in the train phase.
        H_init = data_totinout_train[tspan_train[0],0]
        p_D_bar=res.x[0]
        gamma_D_bar=res.x[1]
        S_bar_init_opt_D_bar = res.x[2] - H_init
        beta_bar_opt_D_bar = res.x[3]
        if show_figures:
             make_plots_D_bar(beta_bar_opt_D_bar,gamma_D_bar,S_bar_init_opt_D_bar,H_init,tspan_train,dates,data_totinout)  
    if show_figures:
        #plt.title("Optimized wrt beta_bar and S_bar_init")
        plt.show(block=False)    
    #fig.savefig('C:/Users/odiao/Desktop/Model Covid19/programme_python/SHR_PA/Code_Python/Dpred_bar_opt.eps')   # save the figure to file
    #plt.close(fig) 
        
        
        
        
        
        
        
        
        
        
        
        
         #Modification des données du nombre de décés
    #epsilon=50
    for epsilon in np.arange(1,100,10):
        v1=[-epsilon/2,epsilon/5,epsilon/5,epsilon/5,epsilon/5,epsilon/5,-epsilon/2] # Début des valeurs pour les jours du dimanche au samedi
        v2=np.repeat(np.array([v1]), 17, axis = 0).flatten() #repeter v1 17 fois cad au dernier samedi du tableau
        v3=[-epsilon/2,epsilon/5,epsilon/5,epsilon/5] #compléter les 4 jours restants
        v=np.hstack((v2,v3)) # Join a sequence of arrays along a new axis. 
        death_modified=death+v
        
        CFR = np.sum(death_modified)/(np.sum(death_modified)+np.sum(new_out))
        p = CFR/(1-CFR)
        S_bar_init_opt=17800
        H_init=370.0
        beta_bar_opt= 1.5e-05 
        N_bar=S_bar_init_opt+H_init
        D_bar=np.cumsum(death_modified)
        D_day=p*gamma*N_bar*(1-np.exp(-(beta_bar_opt*D_bar)/(p*gamma)))-gamma*D_bar
        fig=plt.figure(figsize=(12,6))
        plt.plot(D_bar, death_modified, 'o', markersize=4, label='data')
        plt.plot(D_bar, D_day, label='fitted model')
        plt.xlabel("Cumulative number of fatalities")
        plt.ylabel("Daily fatalities")
        plt.legend()
        plt.show()
    #fig.savefig('C:/Users/odiao/Desktop/Redaction_darticles_Latex/SHR_PA/Figures/fd.pdf')   # save the figure to file
    #plt.close(fig)
    #************Optimization
        def modell(x, u):
            return x[0]*x[1]*x[2]*(1-np.exp(-(x[3]*u)/(x[0]*x[1])))-x[1]*u
        def funn(x, u, y):
            return modell(x, u) - y  
        y=death_modified
        u=np.cumsum(death_modified)  
        x0=[p,gamma, N_bar,beta_bar_opt]
        res = least_squares(funn,x0, args=(u, y), verbose=1)
        u_test = u
        y_test = modell(res.x, u_test)
        fig=plt.figure(figsize=(12,6))
        plt.plot(u, y, 'o', markersize=4, label='data')
        plt.plot(u_test, y_test, label='fitted model')
        plt.xlabel("Cumulative number of fatalities")
        plt.ylabel("Daily fatalities")
        plt.legend()
        plt.show()
    #fig.savefig('C:/Users/odiao/Desktop/Redaction_darticles_Latex/SHR_PA/Figures/fd_opt.pdf')   # save the figure to file
    #plt.close(fig)
        p_D_bar=res.x[0]
        gamma_D_bar=res.x[1]
        S_bar_init_opt_D_bar = res.x[2] - H_init
        beta_bar_opt_D_bar = res.x[3]
    
     #****************************************************************   
         #tester les differentes valeurs
       # Define function for plots:
        def make_plots_D_bar(beta_bar,gamma,S_bar_init,H_init,tspan_train,dates,data_totinout):
            S_bar, H, E, L = simu(beta_bar, gamma, S_bar_init, H_init, tspan=[tspan_train[0],len(total_in)])
            nb_subplots =  show_L + show_D_by_day
            if nb_subplots == 2:
                nb_subplot_rows = 1
                nb_subplot_cols = 2
                plt.rc('xtick', labelsize='x-small')
                plt.rc('ytick', labelsize='x-small')
            else:
                nb_subplot_rows = 1
                nb_subplot_cols = nb_subplots 
            cnt_subplot = 0
            nb_xticks = 4
            dates_ticks = [None] * nb_xticks
            dates_ticks_ind = np.linspace(0,len(total_in)-1,nb_xticks,dtype=int)
            for i in range(0,nb_xticks):
                dates_ticks[i] = dates[dates_ticks_ind[i]]
    
            if show_L:
                cnt_subplot = cnt_subplot + 1
                plt.subplot(nb_subplot_rows,nb_subplot_cols,cnt_subplot)
                if cnt_period == 0:   # assign plot labels    
                    plt.plot(dates,new_out, "-", color='black', label="Discharged", linewidth=1)
                    plt.plot(dates[train_t_start:train_t_end],L[train_t_start:train_t_end],'b--', label="Dis_train")
                    plt.plot(dates[train_t_end-1:test_t_end],L[train_t_end-1:test_t_end],'r-.', label="Dis_pred")
                    plt.legend(fontsize=20)
                else:
                    plt.plot(dates[train_t_start:train_t_end],L[train_t_start:train_t_end],'b--')
                    plt.plot(dates[train_t_end-1:test_t_end],L[train_t_end-1:test_t_end],'r-.')
                plt.xticks(dates_ticks)
                plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
                if cnt_period == nb_periods-1:
                    plt.ylim(bottom=0)
                    
            if show_D_by_day:
                D_by_day=p_D_bar*L
                cnt_subplot = cnt_subplot + 1
                plt.subplot(nb_subplot_rows,nb_subplot_cols,cnt_subplot)
                if cnt_period == 0:   # assign plot labels    
                    plt.plot(dates,death, "-", color='black', label="Deaths", linewidth=1)
                    plt.plot(dates[train_t_start:train_t_end],D_by_day[train_t_start:train_t_end],'b--', label="D_train")
                    plt.plot(dates[train_t_end-1:test_t_end],D_by_day[train_t_end-1:test_t_end],'r-.', label="D_pred")
                    plt.legend(fontsize=20)
                else:
                    plt.plot(dates[train_t_start:train_t_end],D_by_day[train_t_start:train_t_end],'b--')
                    plt.plot(dates[train_t_end-1:test_t_end],D_by_day[train_t_end-1:test_t_end],'r-.')
                plt.xticks(dates_ticks)
                plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
                if cnt_period == nb_periods-1:
                    plt.ylim(bottom=0)
            return
       
        if show_figures:
           fig= plt.figure(figsize=(16,8))
        for cnt_period in range(0, nb_periods):
            # Extract train variables for period cnt_period:
            train_t_start = train_t_start_vals[cnt_period]
            train_t_end = train_t_end_vals[cnt_period]
            test_t_end = test_t_end_vals[cnt_period]
            tspan_train = [train_t_start,train_t_end]
            # The test data is defined to be all the data that occurs from train_t_end.
            # Replace test data by NaN in *_train variables.
            data_totinout_train = copy.deepcopy(data_totinout)  # in order to be able to "hide" entries in data_totinout_train without changing data_totinout
            data_totinout_train[train_t_end:,:] = np.nan  # Beware that data_totinout_train has to be floats.
            # ! Make sure to use only these *_train variables in the train phase.
            H_init = data_totinout_train[tspan_train[0],0]
            p_D_bar=res.x[0]
            gamma_D_bar=res.x[1]
            S_bar_init_opt_D_bar = res.x[2] - H_init
            beta_bar_opt_D_bar = res.x[3]
            if show_figures:
                 make_plots_D_bar(beta_bar_opt_D_bar,gamma_D_bar,S_bar_init_opt_D_bar,H_init,tspan_train,dates,data_totinout)  
        if show_figures:
            #plt.title("Optimized wrt beta_bar and S_bar_init")
            plt.show(block=False)    
        fig.savefig('C:/Users/odiao/Desktop/Redaction_darticles_Latex/SHR_PA/Figures/Dpred_bar_opt.pdf')   # save the figure to file
        plt.close(fig) 
        
        
        
        
        
        
        
        
        
        
        #L1 norm instead of L2 norm for cost function in regression model
        #define a custom cost function (and a convenience wrapper for obtaining the fitted values),
        from scipy.optimize import minimize
        # create wrapper
        def fit(X, params):
            return params[0]*params[1]*params[2]*(1-np.exp(-(params[3]*X)/(params[0]*params[1])))-params[1]*X
        # define cost function (l1-norm)
        def cost_function(params, X, y):
              return np.sum(np.abs(y - fit(X, params)))
         
        y=death
        X = np.cumsum(death)
        params = [p,gamma, N_bar,beta_bar_opt]     
        output = minimize(cost_function, params, args=(X, y))
        y_hat = fit(X, output.x)
        fig = plt.figure(figsize=(18,7))
        plt.subplot(121)
        plt.plot(dates,y, 'o', color='black', label="data")
        plt.plot(dates, y_hat, 'o', color='blue', label="fitted model")
        plt.xlabel("Daily fatalities")
        plt.ylabel("Times/Days")
        plt.legend()
        plt.subplot(122)
        plt.plot(X, y,  'o', color='black', label='data')
        plt.plot(X, y_hat,'o', color='blue', label='fitted model')
        plt.xlabel("Cumulative number of fatalities")
        plt.ylabel("Daily fatalities")
        plt.legend()
        plt.show()
        #fig.savefig('C:/Users/odiao/Desktop/Redaction_darticles_Latex/SHR_PA/Figures/mini.pdf')   # save the figure to file
        #plt.close(fig) 
        
         #tester les differentes valeurs
   # Define function for plots:
    def make_plot(beta_bar,gamma,S_bar_init,H_init,tspan_train,dates,data_totinout):
        S_bar, H, E, L = simu(beta_bar, gamma, S_bar_init, H_init, tspan=[tspan_train[0],len(total_in)])
        nb_subplots =  show_L + show_D_by_day
        if nb_subplots == 2:
            nb_subplot_rows = 1
            nb_subplot_cols = 2
            plt.rc('xtick', labelsize='x-small')
            plt.rc('ytick', labelsize='x-small')
        else:
            nb_subplot_rows = 1
            nb_subplot_cols = nb_subplots 
        cnt_subplot = 0
        nb_xticks = 4
        dates_ticks = [None] * nb_xticks
        dates_ticks_ind = np.linspace(0,len(total_in)-1,nb_xticks,dtype=int)
        for i in range(0,nb_xticks):
            dates_ticks[i] = dates[dates_ticks_ind[i]]

        if show_L:
            cnt_subplot = cnt_subplot + 1
            plt.subplot(nb_subplot_rows,nb_subplot_cols,cnt_subplot)
            if cnt_period == 0:   # assign plot labels    
                plt.plot(dates,new_out, "-", color='black', label="Discharged", linewidth=1)
                plt.plot(dates[train_t_start:train_t_end],L[train_t_start:train_t_end],'b--', label="Dis_train")
                plt.plot(dates[train_t_end-1:test_t_end],L[train_t_end-1:test_t_end],'r-.', label="Dis_pred")
                plt.legend()
            else:
                plt.plot(dates[train_t_start:train_t_end],L[train_t_start:train_t_end],'b--')
                plt.plot(dates[train_t_end-1:test_t_end],L[train_t_end-1:test_t_end],'r-.')
            plt.xticks(dates_ticks)
            plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
            if cnt_period == nb_periods-1:
                plt.ylim(bottom=0)
                
        if show_D_by_day:
            D_by_day=p*L
            cnt_subplot = cnt_subplot + 1
            plt.subplot(nb_subplot_rows,nb_subplot_cols,cnt_subplot)
            if cnt_period == 0:   # assign plot labels    
                plt.plot(dates,death, "-", color='black', label="Deaths", linewidth=1)
                plt.plot(dates[train_t_start:train_t_end],D_by_day[train_t_start:train_t_end],'b--', label="D_train")
                plt.plot(dates[train_t_end-1:test_t_end],D_by_day[train_t_end-1:test_t_end],'r-.', label="D_pred")
                plt.legend()
            else:
                plt.plot(dates[train_t_start:train_t_end],D_by_day[train_t_start:train_t_end],'b--')
                plt.plot(dates[train_t_end-1:test_t_end],D_by_day[train_t_end-1:test_t_end],'r-.')
            plt.xticks(dates_ticks)
            plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
            if cnt_period == nb_periods-1:
                plt.ylim(bottom=0)
        return
   
    if show_figures:
       fig= plt.figure(figsize=(16,8))
    for cnt_period in range(0, nb_periods):
        # Extract train variables for period cnt_period:
        train_t_start = train_t_start_vals[cnt_period]
        train_t_end = train_t_end_vals[cnt_period]
        test_t_end = test_t_end_vals[cnt_period]
        tspan_train = [train_t_start,train_t_end]
        # The test data is defined to be all the data that occurs from train_t_end.
        # Replace test data by NaN in *_train variables.
        data_totinout_train = copy.deepcopy(data_totinout)  # in order to be able to "hide" entries in data_totinout_train without changing data_totinout
        data_totinout_train[train_t_end:,:] = np.nan  # Beware that data_totinout_train has to be floats.
        # ! Make sure to use only these *_train variables in the train phase.
        H_init = data_totinout_train[tspan_train[0],0]
        p=output.x[0]
        gamma=output.x[1]
        S_bar_init_opt = output.x[2] - H_init
        beta_bar_opt = output.x[3]
        if show_figures:
             make_plot(beta_bar_opt,gamma,S_bar_init_opt,H_init,tspan_train,dates,data_totinout)  
    if show_figures:
        #plt.title("Optimized wrt beta_bar and S_bar_init")
        plt.show(block=False)   
        
        
    