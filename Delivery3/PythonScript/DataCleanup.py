'''Joseph Knapp - Data606Capstone

File Info: This file cleans all the data downloaded from the APIDownloads.py file and creates a
    time series dataframe of all the daily: stock prices, risk-free rates, S&P500 market value'''

####################################################################################################
### Libraries 
####################################################################################################

import pandas as pd
import numpy as np
from CommonFunctions import SP500tickers

####################################################################################################
### Common Variables
####################################################################################################

ticker500 = SP500tickers()
startdate = '2010-01-01'

####################################################################################################
### Path to files
####################################################################################################

myPath = 'C:\\Data606\\OriginalData\\'

####################################################################################################
### Join the daily: stock returns, market returns, risk-free rates
####################################################################################################

def SP500Data():
    '''Combines the daily: stock returns, market returs, and risk-free rates'''
    
    #Market data
    market = pd.read_csv(myPath + 'SP500IndexPrice.csv', 
                        index_col = 'Date')
    market = market[market.index > startdate]
    
    #Stock data
    price = pd.read_csv(myPath + 'SP500Pricing.csv',
                         index_col = 'Date')
    price = price[price.index > startdate]    
    
    #Risk-free rates (Tbills)
    r_f = pd.read_csv(myPath + 'TBillRate.csv',
                         index_col = 'Date')
    
    #Merge the dataframes
    price = price.merge(market,
                        left_index = True,
                        right_index = True,
                        how = 'left')
    price = price.merge(r_f,
                        left_index = True,
                        right_index = True,
                        how = 'left')
    
    df = price.fillna(0)
    for i in df.columns:
        if (np.nanstd(df[i][:250]) == 0) or (np.nanstd(df[i][:-10]) == 0):
            price = price.drop(i, axis = 1)

    #Save clean dataframe to file
    price.to_csv('C:\\Data606\\CleanData\\SP500Data.csv')
    
    return price

SP500Data()

