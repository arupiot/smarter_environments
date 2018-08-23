#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 19:48:03 2018

@author: abinaya
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os import listdir
import datetime
from influxdb import DataFrameClient

from iotdesks_occupancy_profiling import *
from utils import *

def train_from_csv(): # --  Train Model using csv -------
    data_path = '/Users/abinaya/USC/Research-ilab/Abinaya/csv/'
    file_name = 'd1motion.csv'
    
    print('### ---------- Running for File: ',file_name)
    data_file_path = data_path + file_name + '/' + file_name
    df = pd.read_csv(data_file_path)
    
    start_date = pd.datetime(2017,02,01) # Start date and End date for Occupancy profiling
    end_date = pd.datetime(2017,03,01)
    
    occupancy_prof = iotdesks_occupancy_profiling(df, start_date, end_date) # Object for the class 
    occupancy_prof.occupancy_profiling()
    #occupancy_prof.df_selected.to_csv('Sample-Results/occupancy_train_'+file_name[:-4]+'.csv')  #save result in csv 
    return occupancy_prof

def train_from_db(iotdesks_client): # --  Train Model using database -------
    start_date_str =  "2018-08-01 00:00:00" 
    end_date_str =  "2018-08-20 00:00:00" 
    
    start_date = pd.to_datetime(start_date_str)
    end_date = pd.to_datetime(end_date_str)
    
    iotdesks_client_data = iotdesks_client.query('select "value" FROM "iotdesks"."autogen"."c1f69ba1_BXK_Motion_Detector" WHERE time > \'' + start_date_str + '\' and time < \'' + end_date_str + '\'')
    
    df = iotdesks_client_data['c1f69ba1_BXK_Motion_Detector']
    df['time'] = df.index
    
    occupancy_prof = iotdesks_occupancy_profiling(df, start_date, end_date) # Object for the class 
    occupancy_prof.occupancy_profiling()
    #occupancy_prof.df_selected.to_csv('Sample-Results/occupancy_train_using_db.csv') #save result in csv
    return occupancy_prof


### First Block -- Database Details -----

host = ""
port = "8086"
user = ""
passwd = ""
db = "iotdesks"
ssl = False

iotdesks_client = DataFrameClient(host=host, port=port, username=user, password=passwd, database=db)

### Second Block -- Train

#occupancy_prof = train_from_csv() #train using data from a csv, either use this or the next line
occupancy_prof = train_from_db(iotdesks_client) #train using data from db, either use this or the previous line

###  Third Block -- Test Model - Real Time ------ 
test_start_date_str =  "2018-08-22 00:00:00" 
test_end_date_str =  "2018-08-22 23:00:00" 

test_start_date = pd.to_datetime(test_start_date_str)
test_end_date = pd.to_datetime(test_end_date_str)

iotdesks_client_data_real_time = iotdesks_client.query('select "value" FROM "iotdesks"."autogen"."c1f69ba1_BXK_Motion_Detector" WHERE time > \'' + test_start_date_str + '\' and time < \'' + test_end_date_str + '\'')

df_real_time = iotdesks_client_data_real_time['c1f69ba1_BXK_Motion_Detector']
df_real_time['time'] = df_real_time.index

df_selected_real_time = preprocess_data(df_real_time, test_start_date, test_end_date) #pre-processing

time_df_real_time, value_df_real_time = feature_extraction(df_selected_real_time, test_start_date, test_end_date, 'value') #feature extraction

pca_fitted_values_real_time = occupancy_prof.pca_model.transform(value_df_real_time) #pca

predicted_pca_fitted_values_real_time = occupancy_prof.gmm.predict(pca_fitted_values_real_time) #gmm

df_selected_real_time = label_each_time_instant(df_selected_real_time, time_df_real_time, predicted_pca_fitted_values_real_time, 'Occupancy_State') #geting labels

#df_selected_real_time.to_csv('Sample-Results/occupancy_test_real_time.csv') #save testing result to csv / can write back to the database







    
