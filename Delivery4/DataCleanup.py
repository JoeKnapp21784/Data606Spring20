'''Joseph Knapp - Data606Capstone

File Info: This file cleans all the data downloaded from the APIDownloads.py file and creates a
    time series dataframe of all the daily: stock prices, risk-free rates, S&P500 market value'''

####################################################################################################
### Libraries 
####################################################################################################

import pandas as pd
import numpy as np

####################################################################################################
### Path to files
####################################################################################################

myPath = 'C:\\Data606\\OriginalData\\'

####################################################################################################
### Common Variables
####################################################################################################

startdate = '2000-01-01'


####################################################################################################
### SP500 Stocks - Join the daily: stock returns, market returns, risk-free rates
####################################################################################################

def StockData():
    '''Combines the daily: stock returns, market returs, and risk-free rates'''
    
    #Data
    SP500 = pd.read_csv(myPath + 'SP500Pricing.csv', index_col = 'Date')
    SP400 = pd.read_csv(myPath + 'SP400Pricing.csv', index_col = 'Date')
    RiskFree = pd.read_csv(myPath + 'TBillRate.csv', index_col = 'Date')
    
    #Remove stocks in more than one indicie
    SP400.drop(set(SP500.columns).intersection(set(SP400.columns)), axis = 1, inplace = True)
    
    #Merge the data
    price = RiskFree.merge(SP500,
                        left_index = True,
                        right_index = True,
                        how = 'left')
    price = price.merge(SP400,
                        left_index = True,
                        right_index = True,
                        how = 'left')
    
    #Get data past a certain date
    price = price[price.index > startdate]
    
    #Remove stocks which do not have consistant data for past 5 years                          
    for i in price.columns:
        if (np.nanstd(price[i].fillna(0)[:-(250*5)]) == 0) or (np.nanstd(price[i].fillna(0)[:-10]) == 0):
            price = price.drop(i, axis = 1)           
    
    price.rename(columns = {'^GSPC':'Market'}, inplace = True)
    
    price = price.reindex(sorted(price.drop(['RiskFree', 'Market'], axis = 1).columns) + ['RiskFree', 'Market'], axis = 1)

    #Save clean dataframe to file
    price.to_csv('C:\\Data606\\CleanData\\StockData.csv')
    
    return price

StockData()



