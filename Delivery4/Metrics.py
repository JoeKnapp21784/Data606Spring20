'''Joseph Knapp - Data606Capstone

File Info: This file holds all the financial functions/calculations required'''

####################################################################################################
### Libraries 
####################################################################################################

import pandas as pd
import numpy as np
from datetime import date, datetime
import random
import matplotlib.pyplot as plt
import seaborn as sns
import time
from CommonFunctions import SP500TickerData, SP400TickerData, FormatData, PercentDone
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

####################################################################################################
### Path to files
####################################################################################################

myPath = 'C:\\Data606\\CleanData\\'

####################################################################################################
### Read in clean stock data
####################################################################################################

StockData = FormatData(myPath + 'StockData.csv')
StockData = StockData.resample('W-MON').mean()
days = list(StockData.index[:-60])


####################################################################################################
### Common Variables
####################################################################################################

#Ticker symbols in S&P500
tickers = list(StockData.columns)[:-2]

df_TD = SP500TickerData().append(SP400TickerData()).reset_index(drop = True)

#Information on the different Tickers(Symbol, Sector, Subsector)
TickerData = pd.DataFrame({'Symbol':tickers}).merge(df_TD, 
                           left_on = 'Symbol',
                           right_on = 'Symbol',
                           how = 'left')



####################################################################################################
### This function checks that a date provided has data and if not provides the date prior
####################################################################################################

def ClosestDate(date):
    '''Returns a valid date if one is not given (weekends, holidays, etc.)'''
    
    return StockData[:date].index[-1]
    
####################################################################################################
### Returns the cost of a quantity of a stock on a given date
####################################################################################################

def StockCost(symbol, date, q = 1):
    '''Finds the price of a stock on a particular date and multiplies it by the quantity'''
    
    return StockData.loc[date].loc[symbol] * q

####################################################################################################
### Expected Return of a Stock
####################################################################################################

def ExpectedAnnualReturn(symbol, date, past_year = False):
    '''Returns the Expected Annual Return of a stock between two dates'''
    if past_year == False:
        return (1 + np.nanmean(StockData[:date][symbol].pct_change())) ** 52
    elif past_year == True:
        date_pr1yr = ClosestDate(date[:-4] + str(int(date[-4:]) - 1))
        return (1 + np.nanmean(StockData[date_pr1yr:date][symbol].pct_change())) ** 52


####################################################################################################
### Finds the Annual Volatility of a stocks return between two dates 
####################################################################################################

def VolatilityAnnualReturn(symbol, date, past_year = False):
    '''Returns the Annual Volatility of a stock between two dates'''
    if past_year == False:
        return np.nanstd(StockData[:date][symbol].pct_change()) * np.sqrt(52)
    elif past_year == True:
        date_pr1yr = ClosestDate(date[:-4] + str(int(date[-4:]) - 1))
        return np.nanstd(StockData[date_pr1yr:date][symbol].pct_change()) * np.sqrt(52)



####################################################################################################
### Finds the Sharpe-ratio of a Stock (Risk-premium over Volatility)
####################################################################################################

def SharpeRatio(symbol, date, past_year = False):
    '''Returns the Sharpe-ratio of the Stock'''
    E = ExpectedAnnualReturn(symbol, date, past_year)
    std = VolatilityAnnualReturn(symbol, date, past_year)
    r_f = StockData['RiskFree'].loc[date]
    return (E - r_f) / std    

####################################################################################################
### Beta of a stock with respect to the Market
####################################################################################################

def BetaStock(symbol, date, past_year = False):
    '''Calculate the Beta of a Stock using the S&P500 as the market portfolio'''    
    if past_year == False:
        df1 = StockData[:date][symbol].pct_change()
        df2 = StockData[:date]['Market'].pct_change()
        cov = df1.cov(df2)
        std_mkt = VolatilityAnnualReturn('Market', date)
        return cov / std_mkt
    elif past_year == True:
        date_pr1yr = ClosestDate(date[:-4] + str(int(date[-4:]) - 1))
        df1 = StockData[date_pr1yr:date][symbol].pct_change()
        df2 = StockData[date_pr1yr:date]['Market'].pct_change()
        cov = df1.cov(df2)
        std_mkt = VolatilityAnnualReturn('Market', date)
        return cov / std_mkt

####################################################################################################
### CAPM of a Stocks
####################################################################################################

def CAPMStock(symbol, date, past_year = False):
    '''Calculates the Return of a stock using CAPM'''    
    B_i = BetaStock(symbol, date, past_year)
    E_mkt = ExpectedAnnualReturn('Market', date, past_year)
    r_f = StockData['RiskFree'].loc[date]
    return r_f + B_i * (E_mkt - r_f)  
   

####################################################################################################
### Alpha of a Stocks
####################################################################################################

def AlphaStock(symbol, date, past_year = False):
    '''Calculates the Alpha of a stock'''    
    return ExpectedAnnualReturn(symbol, date, past_year) - CAPMStock(symbol, date, past_year)

####################################################################################################
### Expected Return given it is negative
####################################################################################################

def TVarLow(symbol, date, past_year = False):
    '''Calculates the expected return given it is negative'''
    if past_year == False:
        df = StockData[:date][symbol].pct_change()
    elif past_year == True:
        date_pr1yr = ClosestDate(date[:-4] + str(int(date[-4:]) - 1))
        df = StockData[date_pr1yr:date][symbol].pct_change()
    return np.mean(df[df < 0])

####################################################################################################
### Expected Return given it is possitive
####################################################################################################

def TVarHigh(symbol, date, past_year = False):
    '''Calculates the expected return given it is negative'''
    if past_year == False:
        df = StockData[:date][symbol].pct_change()
    elif past_year == True:
        date_pr1yr = ClosestDate(date[:-4] + str(int(date[-4:]) - 1))
        df = StockData[date_pr1yr:date][symbol].pct_change()
    return np.mean(df[df > 0])


####################################################################################################
### CDF of Annual Returns
####################################################################################################
'''
AnnualReturnDict = {}
AnnualReturnDict['StockReturn'] = list(pd.read_csv(myPath + 'RandomStockReturn.csv', 
                index_col = [0])['0'])
AnnualReturnDict['MarketReturn'] = list(pd.read_csv(myPath + 'RandomMarketReturn.csv', 
                index_col = [0])['0'])
AnnualReturnDict['ExpectedReturn'] = list(pd.read_csv(myPath + 'RandomStockExpectedAnnualReturn.csv', 
                index_col = [0])['0'])
AnnualReturnDict['SharpeReturn'] = list(pd.read_csv(myPath + 'RandomSharpeReturn.csv', 
                index_col = [0])['0'])
AnnualReturnDict['BetaReturn'] = list(pd.read_csv(myPath + 'RandomBetaReturn.csv', 
                index_col = [0])['0'])
AnnualReturnDict['AlphaReturn'] = list(pd.read_csv(myPath + 'RandomAlphaReturn.csv', 
                index_col = [0])['0'])

all_values = []
plt.figure(figsize = [15,10])
for k, v in AnnualReturnDict.items():
    values, base = np.histogram(v, bins = 1000)
    all_values.extend(v)
    compounded = np.cumsum(values)
    plt.plot(base[:-1], compounded, label = k)
plt.xticks(np.arange(int(min(all_values) * 100 / 25) * 0.25,
                     max(int((max(all_values) * 100 / 25) + 1) * 0.25,3.25),
                     0.25))
plt.yticks(range(0, len(days), int(len(days) / 10)), range(0, 110, 10))
plt.grid()
plt.legend()
plt.show()
'''
####################################################################################################
### CDF of Annual Risk Premium
####################################################################################################
'''
AnnualRiskPremDict = {}
AnnualRiskPremDict['RandomStockRiskPrem'] = list(pd.read_csv(myPath + 'RandomStockRiskPrem.csv', 
                index_col = [0])['0'])
AnnualRiskPremDict['RandomMarketRiskPrem'] = list(pd.read_csv(myPath + 'RandomMarketRiskPrem.csv', 
                index_col = [0])['0'])
AnnualRiskPremDict['ExpectedRiskPrem'] = list(pd.read_csv(myPath + 'RandomStockExpectedAnnualRiskPrem.csv', 
                index_col = [0])['0'])
AnnualRiskPremDict['SharpeRiskPrem'] = list(pd.read_csv(myPath + 'RandomSharpeRiskPrem.csv', 
                index_col = [0])['0'])
AnnualRiskPremDict['BetaRiskPrem'] = list(pd.read_csv(myPath + 'RandomBetaRiskPrem.csv', 
                index_col = [0])['0'])
AnnualRiskPremDict['AlphaRiskPrem'] = list(pd.read_csv(myPath + 'RandomAlphaRiskPrem.csv', 
                index_col = [0])['0'])

all_values = []
plt.figure(figsize = [15,10])
for k, v in AnnualRiskPremDict.items():
    values, base = np.histogram(v, bins = 1000)
    all_values.extend(v)
    compounded = np.cumsum(values)
    plt.plot(base[:-1], compounded, label = k)
plt.xticks(np.arange(int(min(all_values) * 100 / 25) * 0.25,
                     int((max(all_values) * 100 / 25) + 1) * 0.25,
                     0.25))
plt.yticks(range(0, len(days), int(len(days) / 10)), range(0, 110, 10))
plt.grid()
plt.legend()
plt.show()
'''
####################################################################################################
### Percentiles
####################################################################################################
'''
p = np.arange(0,105,5) /100

df_percentile = pd.DataFrame()
df = pd.DataFrame()
df_percentile['Percentile'] = p
for k, v in AnnualReturnDict.items():
    df[k] = v[:995]
    df_percentile[k] = list(df[k].quantile(p))
print(df_percentile)
'''
####################################################################################################
### Data for training and testing
####################################################################################################

#Columns names of data metrics
col = ['Symbol', 'Date', 'StockPrice', 'ExpectedMarketReturn', 'ExpectedMarketReturnPr1Yr', \
       'ExpectedReturn', 'ExpectedReturnPr1Yr', 'Volatility', 'VolatilityPr1Yr', \
       'Sharpe', 'SharpePr1Yr', 'Beta', 'BetaPr1Yr', 'CAPM', 'CAPMPr1Yr', \
       'Alpha', 'AlphaPr1Yr', 'TVarLow', 'TVarLowPr1Yr', 'TVarHigh', 'TVarHighPr1Yr', \
       'Sector', 'SubIndustry',\
       'AnnualReturn', 'AnnualRiskPremium', 'AnnualMarketPremium']

#Create a new file or add to existing one
#df_mod = pd.DataFrame(columns = col)
df_mod = pd.read_csv(myPath + 'ModelingData.csv', index_col = [0])

#Creates the data
i = 0
N = 10000
while i < N: 
    try:
        start = random.choice(days).strftime('%m/%d/%Y')
        start_prior = start[:-4] + str(int(start[-4:]) - 1)
        tickers = list(StockData[start_prior:start].dropna(axis = 1).columns)[:-2]
        stock = random.choice(tickers)
        data = list(StockData[stock].loc[StockData.index > start])[:51]
        data_mkt = list(StockData['Market'].loc[StockData.index > start])[:51]
        row1 = pd.Series([stock, start,
                          StockCost(stock, start),
                          ExpectedAnnualReturn('Market', start),
                          ExpectedAnnualReturn('Market', start, past_year = True),
                          ExpectedAnnualReturn(stock, start),
                          ExpectedAnnualReturn(stock, start, past_year = True),
                          VolatilityAnnualReturn(stock, start),
                          VolatilityAnnualReturn(stock, start, past_year = True),
                          SharpeRatio(stock, start),
                          SharpeRatio(stock, start, past_year = True),
                          BetaStock(stock, start),
                          BetaStock(stock, start, past_year = True),
                          CAPMStock(stock, start),
                          CAPMStock(stock, start, past_year = True),
                          AlphaStock(stock, start),
                          AlphaStock(stock, start, past_year = True),
                          TVarLow(stock, start),
                          TVarLow(stock, start, past_year = True),
                          TVarHigh(stock, start),
                          TVarHigh(stock, start, past_year = True),                          
                          list(TickerData[TickerData['Symbol'] == stock]['Sector'])[0],
                          list(TickerData[TickerData['Symbol'] == stock]['SubIndustry'])[0],
                          data[-1]/data[0], #Annual Return
                          (data[-1]/data[0]) - StockData['RiskFree'][start], #AnnualRiskPrem
                          (data[-1]/data[0]) - (data_mkt[-1]/data_mkt[0])], #MarketPrem
                          
        index = col)
        if ((row1['AnnualReturn'] <= 4) & (row1['ExpectedReturn'] <= 4)):
            df_mod = df_mod.append(row1, ignore_index = True)
           
    except:
        pass
    PercentDone(i, N)
    i += 1

df_mod.dropna(inplace = True)
df_mod.reset_index(drop = True, inplace = True)
df_mod.to_csv(myPath + 'ModelingData.csv')
    



# =============================================================================
# df_mod = pd.read_csv(myPath + 'ModelingData.csv', index_col = [0])
# df_mod95 = df_mod.copy()
# bot = df_mod95.quantile(0.025)
# top = df_mod95.quantile(0.975)
# col_data = df_mod95.columns.drop(['Symbol', 'Date', 'Sector', 'SubIndustry'])
# row = 0
# M = len(col_data)
# while row < len(df_mod95):
#     col = 0
#     while col < M:
#         if not(bot[col_data[col]] <= df_mod95.iloc[row][col_data[col]] < top[col_data[col]]):
#             df_mod95.drop(row, inplace = True, axis = 0)
#             col = len(col_data)
#         else:
#             col += 1
#     row += 1
#     PercentDone(row, len(df_mod95))
# 
# df_mod95 = df_mod95.reset_index(drop = True)   
# df_mod95.to_csv(myPath + 'ModelingData95.csv')
# df_mod95.to_csv(myPath + 'ModelingData95' + str(j) + '.csv')
# 
# j += 1
# =============================================================================














