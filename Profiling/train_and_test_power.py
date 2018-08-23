#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 22:35:36 2018

@author: abinaya
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os import listdir
import datetime
from influxdb import DataFrameClient

from iotdesks_power_profiling import *
from utils import *


### Train Model using csv -------

data_path = '/Users/abinaya/USC/Research-ilab/Abinaya/csv/'

file_name = 'Port16Power.csv'

print('### ---------- Running for File: ',file_name)
data_file_path = data_path + file_name + '/' + file_name
df = pd.read_csv(data_file_path)

start_date = pd.datetime(2017,02,01) # Start date and End date for Power profiling
end_date = pd.datetime(2017,03,01)

power_prof = iotdesks_power_profiling(df, start_date, end_date) # Object for the class 
power_prof.power_profiling()
#occupancy_prof.df_selected.to_csv('Sample-Results/power_train_'+file_name[:-4]+'.csv')

