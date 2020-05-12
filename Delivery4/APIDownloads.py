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
from CommonFunctions import SP500TickerData, PercentDone

####################################################################################################
### Common Variables
####################################################################################################

####################################################################################################
### Path to files
####################################################################################################

myPath = 'C:\\Data606\\OriginalData\\'

####################################################################################################
### Downloads up to date ticker symbols that are in S&P500 from Wikipedia site:
#       https://en.wikipedia.org/wiki/List_of_S%26P_500_companies
####################################################################################################

def SP500TickerData():
    '''Downloads the S&P500 ticker symbols intoo a list'''
    
    table=pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    
    df = table[0]
    df.drop(['SEC filings', 'Headquarters Location', 'CIK', 'Founded'], 
            axis = 1, 
            inplace = True)
    df = df.sort_values(by = ['Symbol']).reset_index(drop = True)
    df.columns = ['Symbol', 'Security', 'Sector', 'SubIndustry', 'DateAdded']
    droplist = set(['BF.B', 'BRK.B', 'TT'])   #these throw errors 
    for i in droplist:
        df = df[df.Symbol != i]
    df.reset_index(drop = True, inplace = True)
    
    return df

SP500Tickers = list(SP500TickerData()['Symbol'])

####################################################################################################
### Get stock pricing data from using Yahoo Finance API
####################################################################################################

def SP500StockDownload():
    '''Downloads the daily historical price data from Yahoo Finance API from 1990 to current.  
    Returns a time-series dataframe of daily adjusted closing prices of all stocks in the S&P500'''
    
    tic = time.time()
    SP500Data = yf.download(tickers = " ".join(SP500Tickers+ ['^GSPC']),
                            start = "1990-01-01")
    SP500Data = SP500Data['Adj Close']    
    SP500Data.to_csv(myPath + 'SP500Pricing.csv')
    print('-----\nSP500HistoricalPrice Download Complete (Time: '\
                                                          + str(int(time.time() - tic))\
                                                          + ' secs)\n-----')        
    return SP500Data

SP500StockDownload()
yf.Ticker('^SPMIDSM').history()
####################################################################################################
### Downloads up to date ticker symbols that are in S&P500 from Wikipedia site:
#       https://en.wikipedia.org/wiki/List_of_S%26P_500_companies
####################################################################################################

def SP400TickerData():
    '''Downloads the S&P400 ticker symbols intoo a list'''
    
    table=pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_400_companies')
    df = table[0]
    df.drop(['SEC filings'], 
            axis = 1, 
            inplace = True)
    df.columns = ['Security', 'Symbol', 'Sector', 'SubIndustry']
    df = df.sort_values(by = ['Symbol']).reset_index(drop = True)
    
    return df

SP400Tickers = list(SP400TickerData()['Symbol'])

####################################################################################################
### Get S&P400 stock pricing data from using Yahoo Finance API
####################################################################################################

def SP400StockDownload():
    '''Downloads the daily historical price data from Yahoo Finance API from 1990 to current.  
    Returns a time-series dataframe of daily adjusted closing prices of all stocks in the S&P500'''
    
    tic = time.time()
    SP400Data = yf.download(tickers = " ".join(SP400Tickers),
                            start = "1990-01-01")
    SP400Data = SP400Data['Adj Close']    
    SP400Data.to_csv(myPath + 'SP400Pricing.csv')
    print('-----\nSP400HistoricalPrice Download Complete (Time: '\
                                                          + str(int(time.time() - tic))\
                                                          + ' secs)\n-----')        
    return SP400Data

SP400StockDownload()

####################################################################################################
### Downloads daily dividends paid to each stock in the S&P500
####################################################################################################

def SP500Dividends():
    '''Downloads S&P500 dividends paid on each stock'''
    
    tic = time.time()
    Div = pd.DataFrame()
    p = 0
    t = SP500Tickers
    for i in t:    
        try:    
            stock = yf.Ticker(i)
            stock = stock.dividends.to_frame().rename(columns = {'Dividends' : i})            
            if Div.empty == True:
                Div = stock
            else:
                Div = Div.merge(stock, 
                                left_index = True,
                                right_index = True,
                                how = 'outer')
        except:
            print(i)            
        if i in t[::int(len(t)*.1)]:
            print(str(p) + '%, ' + '\tTime: ' + str(int(time.time()-tic)) + 'sec')
            p += 10   
    
    Div.fillna(0, inplace = True)
    Div.to_csv(myPath + 'SP500Dividends.csv')
    print('-----\nCompleted in: ' + str(int(time.time()-tic)) + 'sec')
    print('SP500YahooStockDividends Download Complate\n-----')
    
    return SP500Dividends

SP500Dividends()

####################################################################################################
### Downloads daily dividends paid to each stock in the S&P500
####################################################################################################

def SP400Dividends():
    '''Downloads S&P500 dividends paid on each stock'''
    
    tic = time.time()
    Div = pd.DataFrame()
    p = 0
    t = SP400Tickers
    for i in t:    
        try:    
            stock = yf.Ticker(i)
            stock = stock.dividends.to_frame().rename(columns = {'Dividends' : i})            
            if Div.empty == True:
                Div = stock
            else:
                Div = Div.merge(stock, 
                                left_index = True,
                                right_index = True,
                                how = 'outer')
        except:
            print(i)            
        if i in t[::int(len(t)*.1)]:
            print(str(p) + '%, ' + '\tTime: ' + str(int(time.time()-tic)) + 'sec')
            p += 10   
    
    Div.fillna(0, inplace = True)
    Div.to_csv(myPath + 'SP400Dividends.csv')
    print('-----\nCompleted in: ' + str(int(time.time()-tic)) + 'sec')
    print('SP400YahooStockDividends Download Complate\n-----')
    
    return SP400Dividends

SP400Dividends()

###################################################################################################
### Downloads the daily stock splits of each stock in the S&P500
####################################################################################################

def SP500Splits():
    '''Downloads the date and split ratio for stock splits in the S&P500'''
    
    tic = time.time()
    Splits = pd.DataFrame()
    p = 0
    t = SP500Tickers
    for i in t:
        try:    
            stock = yf.Ticker(i)
            stock = stock.splits.to_frame().rename(columns = {'Stock Splits' : i})           
            if Splits.empty == True:
                Splits = stock
            else:
                Splits = Splits.merge(stock,
                                      left_index = True,
                                      right_index = True,how = 'outer'
                                      )
        except:
            print(i)        
        if i in t[::int(len(t)*.1)]:
            print(str(p) + '%, ' + '\tTime: ' + str(int(time.time()-tic)) + 'sec')
            p += 10       
    Splits.fillna(0, inplace = True)
    Splits.to_csv(myPath + 'SP500Splits.csv')
    print('-----\nCompleted in: ' + str(int(time.time()-tic)) + 'sec')
    print('SP500YahooStockSplits Download Complate\n-----')
    
    return SP500Splits

SP500Splits()

###################################################################################################
### Downloads the daily stock splits of each stock in the S&P400
####################################################################################################

def SP400Splits():
    '''Downloads the date and split ratio for stock splits in the S&P500'''
    
    tic = time.time()
    Splits = pd.DataFrame()
    p = 0
    t = SP400Tickers
    for i in t:
        try:    
            stock = yf.Ticker(i)
            stock = stock.splits.to_frame().rename(columns = {'Stock Splits' : i})           
            if Splits.empty == True:
                Splits = stock
            else:
                Splits = Splits.merge(stock,
                                      left_index = True,
                                      right_index = True,how = 'outer'
                                      )
        except:
            print(i)        
        if i in t[::int(len(t)*.1)]:
            print(str(p) + '%, ' + '\tTime: ' + str(int(time.time()-tic)) + 'sec')
            p += 10       
    Splits.fillna(0, inplace = True)
    Splits.to_csv(myPath + 'SP400Splits.csv')
    print('-----\nCompleted in: ' + str(int(time.time()-tic)) + 'sec')
    print('SP400YahooStockSplits Download Complate\n-----')
    
    return SP400Splits

SP400Splits()

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





