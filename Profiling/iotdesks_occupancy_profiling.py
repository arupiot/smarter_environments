#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 10:26:17 2018

@author: abinaya
"""
"""
Class --- iotdesks_occupancy_profiling:
    
    Input: - Pandas Timeseries Dataframe having time and sensor values,
           - Start Time  
           - End Time
    Output: - Occupancy Prediction based on Motion Sensor Data
            - Occupancy Profiling using Kernel Density Estimation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os import listdir
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity

plt.close('all')

class iotdesks_occupancy_profiling:
    
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

    def pca_model(self,value_df):
        # Fit a PCA Model with 2 Components on Value_df dataframe
        n_components = 2
        pca_model = PCA(n_components = n_components)
        pca_model.fit(value_df)
        pca_fitted_values = pca_model.transform(value_df)
        return pca_model, pca_fitted_values
    
    def gmm_model(self, pca_fitted_values):
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
     
    def plot_occupancy_prediction(self, df_selected, feature, prediction_label):
        df_selected_presence_0 = df_selected.loc[df_selected[prediction_label] == self.not_present_label]
        df_selected_presence_1 = df_selected.loc[df_selected[prediction_label] == self.present_label]
        
        plt.figure()        
        plt.plot(df_selected.index, df_selected[feature],label='Motion Sensor')
        plt.scatter(df_selected_presence_0.index, df_selected_presence_0[feature], color='r',label='Not Present')
        plt.scatter(df_selected_presence_1.index, df_selected_presence_1[feature], color='g',label='Present')
        plt.title("Motion Sensor - Clustered")
        plt.legend(loc=1)
    
    def occupancy_prediction(self):
        # Occupancy Prediction Function by calling all the previous function
        # Pre-process the data
        self.preprocess_data()
        print('### -- Done --- Preprocessing')

        # Extract time_df and value_df using feature extraction  
        self.feature = 'value'
        self.time_df, self.value_df = self.feature_extraction(self.feature)
        print('### -- Done --- Feature Extraction')
        
        # PCA Model 
        self.pca_model, self.pca_fitted_values = self.pca_model(self.value_df)
        print('### -- Done --- Principal Component Analysis')
        
        # GMM Model
        self.gmm, self.predicted_pca_fitted_values = self.gmm_model(self.pca_fitted_values)
        print('### -- Done --- Gaussian Mixture Model For Clustering')
        
        # Labelling
        self.prediction_label = 'presence_predicted'
        self.label_each_time_instant(self.time_df, self.predicted_pca_fitted_values ,self.prediction_label)
        print('### -- Done --- Labelling Each Time Instant')
   
        # Finding Present and Not Present Cluster label (Median Hour of the Not Present Cluster label should be lesser)
        df_cluster_0 = self.df_selected.loc[self.df_selected[self.prediction_label] == 0]
        df_cluster_1 = self.df_selected.loc[self.df_selected[self.prediction_label] == 1]
        if df_cluster_0['hour'].median() > df_cluster_1['hour'].median():
            self.present_label = 0
            self.not_present_label = 1
        else:
            self.present_label = 1
            self.not_present_label = 0
        
        #Plotting
        self.plot_occupancy_prediction(self.df_selected, self.feature, self.prediction_label )
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
            df_data_tofit = df.pivot(index='date', columns='hour_min', values='presence_predicted')
            df_data_tofit = df_data_tofit.fillna(0) 
            
            # Fit and get score samples
            self.kde_estimate = pd.DataFrame(index=df_data_tofit.columns, columns=['kde_estimate_0,kde_estimate_1'])
            for i in self.kde_estimate.index:
                kde = KernelDensity(kernel='gaussian', bandwidth=0.75).fit(df_data_tofit[i].values.reshape(-1,1))
                self.kde_estimate.loc[i,'kde_estimate_0'] = np.exp(kde.score_samples(0))[0]
                self.kde_estimate.loc[i,'kde_estimate_1'] = np.exp(kde.score_samples(1))[0]
                 
            #Plot Figures
            fig, axes = plt.subplots(nrows=2, ncols=1)   
            fig.add_subplot(111, frameon=False)
            plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
            self.kde_estimate['kde_estimate_0'].plot(ax=axes[0],label='Not Present - Profile', color='r')
            self.kde_estimate['kde_estimate_1'].plot(ax=axes[1],label='Present - Profile', color='g')
            plt.suptitle('Occupancy Profile (Motion Sensor) - Day of week: ' + self.weekday_dict[weekday_index])
            #plt.xlabel('Time of Day')
            plt.ylabel('Probability - estimate')
            axes[0].legend(loc=1)
            axes[1].legend(loc=1)
        print('### -- Done --- Kernel Density Estimation For Profiling')

      
    def occupancy_profiling(self):
        self.occupancy_prediction()
        self.separate_day_wise_data()
        self.kernel_density_estimation()

            
          
def main():
    ### Path and data file name
    data_path = '/Users/abinaya/USC/Research-ilab/Abinaya/csv/'
    #data_list = listdir(data_path)
    
    # Run for a single file or multiple files
    file_name_list = ['d1motion.csv']
    
    for file_name in file_name_list:
        print('### ---------- Running for File: ',file_name)
        data_file_path = data_path + file_name + '/' + file_name
        df = pd.read_csv(data_file_path)
        
        ### Start date and End date for Occupancy profiling
        start_date = pd.datetime(2017,02,01)
        end_date = pd.datetime(2017,03,01)
        
        ### Object for the class 
        occupancy_prof = iotdesks_occupancy_profiling(df, start_date, end_date)
        occupancy_prof.occupancy_profiling()


if __name__ == "__main__":
    main()