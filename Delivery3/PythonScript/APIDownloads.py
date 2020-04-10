'''Joseph Knapp - Data606Capstone

File Info: This file downloads all the raw data for the project using various API's'''

####################################################################################################
### Libraries 
####################################################################################################

import pandas as pd
import yfinance as yf   #Yahoo Finance API
#import datetime
#import json
import time
from fredapi import Fred   #Federal Reserve Economic Database API
from CommonFunctions import SP500tickers

####################################################################################################
### Common Variables
####################################################################################################

ticker500 = SP500tickers()

####################################################################################################
### Path to files
####################################################################################################

myPath = 'C:\\Data606\\OriginalData\\'

####################################################################################################
### Get stock pricing data from using Yahoo Finance API
####################################################################################################

def SP500YahooPriceDownload():
    '''Downloads the daily historical price data from Yahoo Finance API from 1990 to current.  
    Returns a time-series dataframe of daily adjusted closing prices of all stocks in the S&P500'''
    
    tic = time.time()
    SP500Data = yf.download(tickers = " ".join(ticker500),
                            start = "1990-01-01")
    SP500Data = SP500Data['Adj Close']    
    SP500Data.to_csv(myPath + 'SP500Pricing.csv')
    print('-----\nSP500HistoricalPrice Download Complete (Time: '\
                                                          + str(int(time.time() - tic))\
                                                          + ' secs)\n-----')        
    return SP500Data

SP500YahooPriceDownload()

####################################################################################################
### Get the overall S&P500 Index historical data
####################################################################################################

def SP500IndexYahooDownload():
    '''Downloads the S&P500 Overall Market histrical data'''
    
    SP500IndexPrice = yf.Ticker('^GSPC').history(start = '1990-01-01')[['Close']]
    SP500IndexPrice.columns = ['Market']
    SP500IndexPrice.to_csv(myPath + 'SP500IndexPrice.csv')
    print('-----\nSP500IndexPrice Download Complete\n-----')
    
    return SP500IndexPrice

SP500IndexYahooDownload()

####################################################################################################
### Downloads daily dividends paid to each stock in the S&P500
####################################################################################################

def SP500Dividends():
    '''Downloads S&P500 dividends paid on each stock'''
    
    tic = time.time()
    SP500Dividends = pd.DataFrame()
    p = 0
    t = ticker500
    for i in t:    
        try:    
            stock = yf.Ticker(i)
            stock = stock.dividends.to_frame().rename(columns = {'Dividends' : i})            
            if SP500Dividends.empty == True:
                SP500Dividends = stock
            else:
                SP500Dividends = SP500Dividends.merge(stock,
                                                      left_index = True,
                                                      right_index = True,
                                                      how = 'outer')
        except:
            print(i)            
        if i in t[::int(len(t)*.1)]:
            print(str(p) + '%, ' + '\tTime: ' + str(int(time.time()-tic)) + 'sec')
            p += 10   
    
    SP500Dividends.fillna(0, inplace = True)
    SP500Dividends.to_csv(myPath + 'SP500Dividends.csv')
    print('-----\nCompleted in: ' + str(int(time.time()-tic)) + 'sec')
    print('SP500YahooStockDividends Download Complate\n-----')
    
    return SP500Dividends

SP500Dividends()

####################################################################################################
### Downloads the daily stock splits of each stock in the S&P500
####################################################################################################

def SP500Splits():
    '''Downloads the date and split ratio for stock splits in the S&P500'''
    
    tic = time.time()
    SP500Splits = pd.DataFrame()
    p = 0
    t = ticker500
    for i in t:
        try:    
            stock = yf.Ticker(i)
            stock = stock.splits.to_frame().rename(columns = {'Stock Splits' : i})           
            if SP500Splits.empty == True:
                SP500Splits = stock
            else:
                SP500Splits = SP500Splits.merge(stock,
                                                left_index = True,
                                                right_index = True,
                                                how = 'outer')
        except:
            print(i)        
        if i in t[::int(len(t)*.1)]:
            print(str(p) + '%, ' + '\tTime: ' + str(int(time.time()-tic)) + 'sec')
            p += 10       
    SP500Splits.fillna(0, inplace = True)
    SP500Splits.to_csv(myPath + 'SP500Splits.csv')
    print('-----\nCompleted in: ' + str(int(time.time()-tic)) + 'sec')
    print('SP500YahooStockSplits Download Complate\n-----')
    
    return SP500Splits

SP500Splits()

####################################################################################################
### Downloads the daily 10-year Treasury Bill rate using the Federal Reserve Economic Database
#       (FRED) API
####################################################################################################

def DailyRiskFreeRates():
    
    fred = Fred(api_key='5e8edcc00dd0a9e40375540f32696316')
    TBills = fred.get_series('DGS10').to_frame().reset_index()
    TBills.columns = ['Date', 'RiskFree']
    TBills.set_index('Date', inplace = True)
    TBills['RiskFree'] =  TBills['RiskFree'].apply(lambda x: x / 100)
    TBills.to_csv(myPath + 'TBillRate.csv')
    
    return TBills

DailyRiskFreeRates()





