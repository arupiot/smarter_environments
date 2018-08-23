#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May 21 21:10:07 2018

@author: abinaya
"""
"""
Class --- iotdesks_power_profiling:
    
    Input: - Pandas Timeseries Dataframe having time and laptop charger power values,
           - Start Time  
           - End Time
    Output: - Power Usage Prediction based on Laptop Charger Power Consumption Data
            - Power Usage Profiling using Kernel Density Estimation
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os import listdir
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity

plt.close('all')

class iotdesks_power_profiling:
    
    def __init__(self,df,start_date,end_date):
        # Pass the raw data, start date and end date while initializing the object for the class
        self.df = df
        self.start_date = start_date
        self.end_date = end_date
    
    def preprocess_data(self):
        # Preprocess the raw data, Convert time and value column to numeric data
        self.df['time'] = pd.to_numeric(self.df['time'], errors='coerce')
        self.df['value'] = pd.to_numeric(self.df['value'], errors='coerce')
        self.df['time'] = pd.to_datetime(self.df['time']/10**6, unit='ms')
        self.df.index = self.df['time']
        
        # Select the required data from start date to end date
        self.df_selected = self.df.loc[(self.df.index > self.start_date) & (self.df.index < self.end_date)]
        
        # Select one data point for every minute - If more than one datapoints exist within a minute
        self.df_selected['time'] = self.df_selected['time'].map(lambda t: t.strftime('%Y-%m-%d %H:%M'))
        self.df_selected = self.df_selected.sort_values('value', ascending=False).drop_duplicates('time').sort_index().reset_index(drop=True)
        self.df_selected['time'] = pd.to_datetime(self.df_selected['time'])
        self.df_selected['dayofweek'] = pd.Series(self.df_selected['time']).dt.dayofweek
        self.df_selected['hour'] = self.df_selected['time'].dt.hour
        self.df_selected.index = self.df_selected['time']
        
        # Get Rolling mean of n=30 data points to get smoothed/averaged data
        self.df_selected['value_rollingmean'] = pd.rolling_mean(self.df_selected['value'],30)
    
    def feature_extraction(self, feature):
        # Convert time series data to rows and columns of feature values - To be able to be clustered
        # Initialize two dataframes for value and time
        freq_mins = '6min'
        time_df = pd.DataFrame()
        value_df = pd.DataFrame()
        
        # Get time start and end for every row for convenience
        feature_vector_start_dates = pd.date_range(self.start_date, self.end_date, freq=freq_mins)
        
        count = 0
        for i in range(0,len(feature_vector_start_dates)-1):
            temp_date_range = pd.date_range(feature_vector_start_dates[i],feature_vector_start_dates[i+1],freq='1min')
            for j in range(0,len(temp_date_range)):
                time_df.loc[count,j] = temp_date_range[j]
                if temp_date_range[j] in self.df_selected.index:
                    value_df.loc[count,j] = self.df_selected.loc[temp_date_range[j],feature] 
                else:
                    value_df.loc[count,j] = 0
            count += 1
        
        value_df = value_df.fillna(0)
        return time_df, value_df

    def return_pca_model(self,value_df):
        # Fit a PCA Model with 2 Components on Value_df dataframe
        n_components = 2
        pca_model = PCA(n_components = n_components)
        pca_model.fit(value_df)
        pca_fitted_values = pca_model.transform(value_df)
        return pca_model, pca_fitted_values
    
    def return_gmm_model(self, pca_fitted_values):
        # Fit a GMM model with 2 clusteres
        n_components = 2
        gmm = GaussianMixture(n_components=n_components)
        gmm.fit(pca_fitted_values)
        predicted_pca_fitted_values = gmm.predict(pca_fitted_values)
        return gmm, predicted_pca_fitted_values
    
    def label_each_time_instant(self, time_df, predicted_pca_fitted_values, prediction_label):
        # Label time instants of each row to 0 and 1 Cluster   
        predicted_1_df = time_df[predicted_pca_fitted_values == 1]
        predicted_1_time = np.unique(np.array(predicted_1_df[range(0,7)]).ravel())
        self.df_selected[prediction_label] = 0
        self.df_selected.loc[self.df_selected.index.isin(list(predicted_1_time)),prediction_label] = 1
     
    def plot_power_prediction(self, df_selected, feature, prediction_label):
        df_selected_presence_0 = df_selected.loc[df_selected[prediction_label] == 0]
        df_selected_presence_1 = df_selected.loc[df_selected[prediction_label] == 1]
        df_selected_presence_2 = df_selected.loc[df_selected[prediction_label] == 2]      
        
        plt.figure()        
        plt.plot(df_selected.index, df_selected[feature],label='Laptop Charger Power(Averaged)')
        plt.scatter(df_selected_presence_0.index, df_selected_presence_0[feature], color='r',label='OFF')
        plt.scatter(df_selected_presence_1.index, df_selected_presence_1[feature], color='k',label='Standby')
        plt.scatter(df_selected_presence_2.index, df_selected_presence_2[feature], color='g',label='ON')
        plt.title("Laptop Charger - Clustered")
        plt.legend(loc=1)

    
    def power_prediction(self):
        # Pre-process the data
        self.preprocess_data()
        print('### -- Done --- Preprocessing')
        
        # Extract time_df and value_df using feature extraction
        self.feature1 = 'value_rollingmean'
        self.time_df1, self.value_df1 = self.feature_extraction(self.feature1)
        self.time_df1 = self.time_df1.loc[~(self.value_df1==0).all(axis=1)]
        self.value_df1 = self.value_df1.loc[~(self.value_df1==0).all(axis=1)]
        print('### -- Done --- Feature Extraction - First Level')

        # PCA Model 
        self.pca_model1, self.pca_fitted_values1 = self.return_pca_model(self.value_df1)
        print('### -- Done --- Principal Component Analysis - First Level')
       
        # GMM Model
        self.gmm1, self.predicted_pca_fitted_values1 = self.return_gmm_model(self.pca_fitted_values1)
        print('### -- Done --- Gaussian Mixture Model For Clustering - First Level')
        
        # Labelling
        self.prediction_label1 = 'hibernation_predicted_1'
        self.label_each_time_instant(self.time_df1, self.predicted_pca_fitted_values1 ,self.prediction_label1)
        print('### -- Done --- Labelling Each Time Instant - First Level')
        
        # Copy first dataframe to a new one 
        self.df_selected1 = self.df_selected.copy()
        
        # Finding OFF and Remaining Cluster label (Median Hour of the OFF Cluster label should be lesser)
        df_first_cluster_0 = self.df_selected1.loc[self.df_selected1[self.prediction_label1] == 0]
        df_first_cluster_1 = self.df_selected1.loc[self.df_selected1[self.prediction_label1] == 1]
        if df_first_cluster_0['hour'].median() > df_first_cluster_1['hour'].median():
            self.label_for_next_cluster = 0
            self.off_label = 1
        else:
            self.label_for_next_cluster = 1
            self.off_label = 0
           
        # Select Dataframe for second clustering
        self.df_selected = self.df_selected[self.df_selected[self.prediction_label1] == self.label_for_next_cluster]
        
        # Extract time_df and value_df using feature extraction -- 2
        self.feature2 = 'value_rollingmean'
        self.time_df2, self.value_df2 = self.feature_extraction(self.feature2)
        self.time_df2 = self.time_df2.loc[~(self.value_df2==0).all(axis=1)]
        self.value_df2 = self.value_df2.loc[~(self.value_df2==0).all(axis=1)]
        print('### -- Done --- Feature Extraction - Second Level')
        
        # PCA Model -- 2
        self.pca_model2, self.pca_fitted_values2 = self.return_pca_model(self.value_df2)
        print('### -- Done --- Principal Component Analysis - Second Level')
        
        # GMM Model -- 2
        self.gmm2, self.predicted_pca_fitted_values2 = self.return_gmm_model(self.pca_fitted_values2)
        print('### -- Done --- Gaussian Mixture Model For Clustering - Second Level')
       
        # Labelling -- 2
        self.prediction_label2 = 'hibernation_predicted_2'
        self.label_each_time_instant(self.time_df2, self.predicted_pca_fitted_values2 ,self.prediction_label2)
        print('### -- Done --- Labelling Each Time Instant - Second Level')

        # Copy second dataframe to a new one
        self.df_selected2 = self.df_selected.copy()

        # Finding Standby and ON Cluster label (Median Hour of the STANDBY Cluster label should be lesser)
        df_second_cluster_0 = self.df_selected2.loc[self.df_selected2[self.prediction_label2] == 0]
        df_second_cluster_1 = self.df_selected2.loc[self.df_selected2[self.prediction_label2] == 1]
        if df_second_cluster_0['hour'].median() > df_second_cluster_1['hour'].median():
            self.standby_label = 1
            self.on_label = 0
        else:
            self.standby_label = 0
            self.on_label = 1
 
        # Combining
        self.feature = 'value_rollingmean'
        self.prediction_label = 'hibernation_predicted'
        self.df_selected = self.df_selected1.copy()
        self.df_selected[self.prediction_label2] = self.df_selected2[self.prediction_label2]
        self.df_selected.loc[self.df_selected[self.prediction_label1] == self.off_label, self.prediction_label] = 0
        self.df_selected.loc[self.df_selected[self.prediction_label2] == self.standby_label, self.prediction_label] = 1
        self.df_selected.loc[self.df_selected[self.prediction_label2] == self.on_label, self.prediction_label] = 2
        print('### -- Done --- Combining First and Second Level Outputs')
        
        #Plotting
        self.plot_power_prediction(self.df_selected, self.feature, self.prediction_label )
        print('### -- Done --- Plotting')
   
    def separate_day_wise_data(self):
        # Seperate a single dictionary with 7 different dataframes for every day of the week
        self.df_selected_dict = {}
        self.df_selected_dict['df_selected_0'] = self.df_selected.loc[self.df_selected['dayofweek'] == 0]
        self.df_selected_dict['df_selected_1'] = self.df_selected.loc[self.df_selected['dayofweek'] == 1]
        self.df_selected_dict['df_selected_2'] = self.df_selected.loc[self.df_selected['dayofweek'] == 2]
        self.df_selected_dict['df_selected_3'] = self.df_selected.loc[self.df_selected['dayofweek'] == 3]
        self.df_selected_dict['df_selected_4'] = self.df_selected.loc[self.df_selected['dayofweek'] == 4]
        self.df_selected_dict['df_selected_5'] = self.df_selected.loc[self.df_selected['dayofweek'] == 5]
        self.df_selected_dict['df_selected_6'] = self.df_selected.loc[self.df_selected['dayofweek'] == 6]
  
    def kernel_density_estimation(self):
        self.weekday_dict = {0:'Monday',1:'Tuesday',2:'Wednesday',3:'Thursday',4:'Friday',5:'Saturday',6:'Sunday'}

        for weekday_index in range(0,7):
            # Get data for every day of the week
            df = self.df_selected_dict['df_selected_'+str(weekday_index)]
            df['date'] = df['time'].dt.date
            df['hour_min'] = df['time'].map(lambda t: t.strftime('%H:%M'))
            
            # Change the dataframe to required format to fit the Kernel Density Estimate
            df_data_tofit = df.pivot(index='date', columns='hour_min', values=self.prediction_label)
            df_data_tofit = df_data_tofit.fillna(0)
            
            # Fit and get score samples
            self.kde_estimate = pd.DataFrame(index=df_data_tofit.columns, columns=['kde_estimate_0,kde_estimate_1,kde_estimate_2'])
            for i in self.kde_estimate.index:
                kde = KernelDensity(kernel='gaussian', bandwidth=0.75).fit(df_data_tofit[i].values.reshape(-1,1))
                self.kde_estimate.loc[i,'kde_estimate_0'] = np.exp(kde.score_samples(0))[0]
                self.kde_estimate.loc[i,'kde_estimate_1'] = np.exp(kde.score_samples(1))[0]
                self.kde_estimate.loc[i,'kde_estimate_2'] = np.exp(kde.score_samples(2))[0]

            #Plot Figures
            fig, axes = plt.subplots(nrows=3, ncols=1)   
            fig.add_subplot(111, frameon=False)
            plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
            self.kde_estimate['kde_estimate_2'].plot(ax=axes[0],label='ON - Profile', color='g')
            self.kde_estimate['kde_estimate_1'].plot(ax=axes[1],label='Standby - Profile', color='k')
            self.kde_estimate['kde_estimate_0'].plot(ax=axes[2],label='OFF - Profile', color='r')
            plt.suptitle('Power Usage Profile (Laptop Charger) - Day of week: ' + self.weekday_dict[weekday_index])
            #plt.xlabel('Time of Day')
            plt.ylabel('Probability - estimate')
            axes[0].legend(loc=1)
            axes[1].legend(loc=1)
            axes[2].legend(loc=1)
        print('### -- Done --- Kernel Density Estimation For Profiling')
            
    def power_profiling(self):
        self.power_prediction()
        self.separate_day_wise_data()
        self.kernel_density_estimation() 
        plt.show()

