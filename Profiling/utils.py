#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 19:36:08 2018

@author: abinaya
"""
import numpy as np
import pandas as pd

def preprocess_data(df, start_date, end_date):
    # Preprocess the raw data, Convert time and value column to numeric data
    df['time'] = pd.to_numeric(df['time'], errors='coerce')
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    df['time'] = pd.to_datetime(df['time']/10**6, unit='ms')
    df.index = df['time']
    
    # Select the required data from start date to end date
    df_selected = df.loc[(df.index > start_date) & (df.index < end_date)]
    
    # Select one data point for every minute - If more than one datapoints exist within a minute
    df_selected['time'] = df_selected['time'].map(lambda t: t.strftime('%Y-%m-%d %H:%M'))
    df_selected = df_selected.sort_values('value', ascending=False).drop_duplicates('time').sort_index().reset_index(drop=True)
    df_selected['time'] = pd.to_datetime(df_selected['time'])
    df_selected['dayofweek'] = pd.Series(df_selected['time']).dt.dayofweek
    df_selected['hour'] = df_selected['time'].dt.hour
    df_selected.index = df_selected['time']
    return df_selected

def feature_extraction(df_selected, start_date, end_date, feature):
    # Convert time series data to rows and columns of feature values - To be able to be clustered
    # Initialize two dataframes for value and time
    freq_mins = '6min'
    time_df = pd.DataFrame()
    value_df = pd.DataFrame()
    
    # Get time start and end for every row for convenience
    feature_vector_start_dates = pd.date_range(start_date, end_date, freq=freq_mins)
    
    count = 0
    for i in range(0,len(feature_vector_start_dates)-1):
        temp_date_range = pd.date_range(feature_vector_start_dates[i],feature_vector_start_dates[i+1],freq='1min')
        for j in range(0,len(temp_date_range)):
            time_df.loc[count,j] = temp_date_range[j]
            if temp_date_range[j] in df_selected.index:
                value_df.loc[count,j] = df_selected.loc[temp_date_range[j],feature] 
            else:
                value_df.loc[count,j] = 0
        count += 1
    
    value_df = value_df.fillna(0)
    return time_df, value_df

def label_each_time_instant(df_selected, time_df, predicted_pca_fitted_values, prediction_label):
    # Label time instants of each row to 0 and 1 Cluster   
    predicted_1_df = time_df[predicted_pca_fitted_values == 1]
    predicted_1_time = np.unique(np.array(predicted_1_df[range(0,7)]).ravel())
    df_selected[prediction_label] = 0
    df_selected.loc[df_selected.index.isin(list(predicted_1_time)),prediction_label] = 1
    return df_selected
 

