'''Joseph Knapp - Data606Capstone

File Info: This file holds all the financial functions/calculations required'''

####################################################################################################
### Libraries 
####################################################################################################

import pandas as pd
import numpy as np
from datetime import date, timedelta
import random
import matplotlib.pyplot as plt
import seaborn as sns
import time
from CommonFunctions import SP500tickers, FormatData

####################################################################################################
### Path to files
####################################################################################################

myPath = 'C:\\Data606\\CleanData\\'

####################################################################################################
### Read in clean stock data
####################################################################################################

SP500 = FormatData(myPath + 'SP500Data.csv')

####################################################################################################
### Common Variables
####################################################################################################

ticker500 = list(SP500.columns)[:-2]    #stock symbols in S&P with valid data
days = random.sample(list(SP500.index[500:-300]), 550)    #days in data frame with stock data 

####################################################################################################
### This function checks that a date provided has data and if not provides the date prior
####################################################################################################

def ClosestDate(date):
    '''Returns a valid date if one is not given (weekends, holidays, etc.)'''
    
    d = list(SP500[SP500.index <= date].index)[-1]
    
    return d

####################################################################################################
### Returns the cost of a quantity of a stock on a given date
####################################################################################################

def StockCost(symbol, date, q = 1):
    '''Finds the price of a stock on a particular date and multiplies it by the quantity'''
    
    return SP500.loc[date].loc[symbol] * q

####################################################################################################
### Expected Return of a Stock
####################################################################################################

def ExpectedAnnualReturn(symbol, end, start = SP500.index[0]):
    '''Returns the Expected Annual Return of a stock between two dates'''
    
    df = SP500[symbol].pct_change()
    stock = np.array(df[(df.index < end) & (df.index >= start)])
    E_da = np.nanmean(stock)
    E_an = E_da * 250
    
    return E_an

####################################################################################################
### Finds the Annual Volatility of a stocks return between two dates 
####################################################################################################

def VolatilityAnnualReturn(symbol, end, start = SP500.index[0]):
    '''Returns the Annual Volatility of a stock between two dates'''
    
    df = SP500[symbol].pct_change()
    stock = np.array(df[(df.index < end) & (df.index >= start)])
    std_da = np.nanstd(stock)
    std_an = std_da * np.sqrt(250)

    return std_an

####################################################################################################
### Finds the Sharpe-ratio of a Stock (Risk-premium over Volatility)
####################################################################################################

def SharpeRatio(symbol, end):
    '''Returns the Sharpe-ratio of the Stock'''
    
    E = ExpectedAnnualReturn(symbol, end)
    r_f = SP500['RiskFree'].loc[end]
    std = VolatilityAnnualReturn(symbol, end)
    if std == 0:
        s = 0
    else:
        s = (E - r_f) / std
    
    return s

####################################################################################################
### Beta of a stock with respect to the Market
####################################################################################################

def BetaStock(symbol, date):
    '''Calculate the Beta of a Stock using the S&P500 as the Efficiient Market'''
    
    d = ClosestDate(date)
    cov = SP500[SP500.index < d][symbol].cov(SP500[SP500.index < d]['Market'])
    mkt = SP500[SP500.index < d]['Market'].var()
    B = cov / mkt

    return B

####################################################################################################
### CAPM of a Stocks
####################################################################################################

def CAPMStock(symbol, end, start = SP500.index[0]):
    '''Calculates the Return of a stock using CAPM'''
    
    B_i = BetaStock(symbol, end)
    E_mkt = ExpectedAnnualReturn('Market', end, start)
    r_f = SP500['RiskFree'].loc[end]
    r_i = r_f + B_i * (E_mkt - r_f)
    
    return r_i

####################################################################################################
### Buying Random data on random day - baseline return data
####################################################################################################

def RandomStockBaselineAnnualReturn():
    
    start = random.choice(days).strftime('%m/%d/%Y')
    end = ClosestDate(start[:6] + str(int(start[-4:]) + 1))
    s = random.choice(ticker500)               
    df1 = SP500[s].loc[(SP500.index >= ClosestDate(start)) & \
              (SP500.index <= ClosestDate(end))]
    df1.resample('D')
    stock = np.array((df1.pct_change() + 1).cumprod())

    return stock

tic = time.time()
i = 0
N = len(days)
df_BaseReturn = pd.DataFrame()
for x in days:
    i += 1
    print(i)
    try:
        df_BaseReturn[x] = RandomStockBaselineAnnualReturn()[:250]
    except:
        pass
    print(int(i / N * 100))
print(str(int(time.time() - tic)) + ' secs')
df_BaseReturn.to_csv(myPath + 'BaseReturn.csv')


df_BaseReturn = pd.read_csv(myPath + 'MBaseReturn.csv', index_col = [0])
plt.figure(figsize=[15, 10])
plt.plot(df_BaseReturn - 1, linewidth = 1, c = 'b', alpha = 0.1)
plt.plot((df_BaseReturn - 1).quantile([0.05, 0.25, 0.5, 0.75, 0.95], axis = 1).T, linewidth = 3, c = 'r')
plt.grid(axis = 'y')
plt.title('Random Stock (Baseline) - Annual Return')
plt.xlabel('Time (One Year = 250 trading days)')
plt.ylabel('Annual Return (%)')
plt.show()


def RandomStockBaselineAnnualPremium():
        
    start = random.choice(days).strftime('%m/%d/%Y')
    end = ClosestDate(start[:6] + str(int(start[-4:]) + 1))
    s = random.choice(ticker500)               
    df1 = SP500[s].loc[(SP500.index >= ClosestDate(start)) & \
              (SP500.index <= ClosestDate(end))]
    df1.resample('D')
    stock = np.array((df1.pct_change() + 1).cumprod())
    r_f = (np.array([(1 + SP500['RiskFree'].loc[start]) ** (1/250)] * len(stock))).cumprod()
    riskPrem = stock - r_f
    
    return riskPrem

tic = time.time()
i = 0
N = len(days)
df_BasePrem = pd.DataFrame()
for x in days:
    i += 1
    try:
        df_BasePrem[x] = RandomStockBaselineAnnualPremium()[:250]
    except:
        pass
    print(int(i / N * 100))
print(str(int(time.time() - tic)) + ' secs')
df_BasePrem.to_csv(myPath + 'BasePremium.csv')

df_BasePrem = pd.read_csv(myPath + 'BasePremium.csv', index_col = [0])
plt.figure(figsize=[15, 10])
plt.plot(df_BasePrem, linewidth = 1, c = 'b', alpha = 0.1)
plt.plot(df_BasePrem.quantile([0.05, 0.25, 0.5, 0.75, 0.95], axis = 1).T, linewidth = 3, c = 'r')
plt.grid(axis = 'y')
plt.title('Random Stock (Baseline) - Annual Premium')
plt.xlabel('Time (One Year = 250 trading days)')
plt.ylabel('Annual Return (%)')
plt.show()

####################################################################################################
### Investing in Market on a random day
####################################################################################################

def RandomMarketBaselineAnnualReturn():
    
    start = random.choice(days).strftime('%m/%d/%Y')
    end = ClosestDate(start[:6] + str(int(start[-4:]) + 1))              
    df1 = SP500['Market'].loc[(SP500.index >= ClosestDate(start)) & \
              (SP500.index <= ClosestDate(end))]
    df1.resample('D')
    stock = np.array((df1.pct_change() + 1).cumprod())

    return stock

tic = time.time()
i = 0
N = len(days)
df_MktReturn = pd.DataFrame()
for x in days:
    i += 1
    try:
        df_MktReturn[x] = RandomMarketBaselineAnnualReturn()[:250]
    except:
        pass
    print(int(i / N * 100))
print(str(int(time.time() - tic)) + ' secs')
df_MktReturn.to_csv(myPath + 'MarketReturn.csv')


df_MktReturn = pd.read_csv(myPath + 'MarketReturn.csv', index_col = [0])
plt.figure(figsize=[15, 10])
plt.plot(df_MktReturn - 1, linewidth = 1, c = 'b', alpha = 0.1)
plt.plot((df_MktReturn - 1).quantile([0.05, 0.25, 0.5, 0.75, 0.95], axis = 1).T, linewidth = 3, c = 'r')
plt.grid(axis = 'y')
plt.title('Random Market Return (Baseline) - Annual Return')
plt.xlabel('Time (One Year = 250 trading days)')
plt.ylabel('Annual Return (%)')
plt.show()

def RandomMarketBaselineAnnualPremium():
        
    start = random.choice(days).strftime('%m/%d/%Y')
    end = ClosestDate(start[:6] + str(int(start[-4:]) + 1))        
    df1 = SP500['Market'].loc[(SP500.index >= ClosestDate(start)) & \
              (SP500.index <= ClosestDate(end))]
    df1.resample('D')
    stock = np.array((df1.pct_change() + 1).cumprod())
    r_f = (np.array([(1 + SP500['RiskFree'].loc[start]) ** (1/250)] * len(stock))).cumprod()
    riskPrem = stock - r_f
    
    return riskPrem

tic = time.time()
i = 0
N = len(days)
df_MktPrem = pd.DataFrame()
for x in days:
    i += 1
    try:
        df_MktPrem[x] = RandomMarketBaselineAnnualPremium()[:250]
    except:
        pass
    print(int(i / N * 100))
print(str(int(time.time() - tic)) + ' secs')
df_MktPrem.to_csv(myPath + 'MarketPremium.csv')

df_MktPrem = pd.read_csv(myPath + 'MarketPremium.csv', index_col = [0])
plt.figure(figsize=[15, 10])
plt.plot(df_MktPrem, linewidth = 1, c = 'b', alpha = 0.1)
plt.plot(df_MktPrem.quantile([0.05, 0.25, 0.5, 0.75, 0.95], axis = 1).T, linewidth = 3, c = 'r')
plt.grid(axis = 'y')
plt.title('Random Market (Baseline) - Annual Premium')
plt.xlabel('Time (One Year = 250 trading days)')
plt.ylabel('Annual Return (%)')
plt.show()


####################################################################################################
### Buying Stock based on Expected Return only
####################################################################################################

def RandsomStockAnnualReturn():
    
    start = random.choice(days).strftime('%m/%d/%Y')
    end = ClosestDate(start[:6] + str(int(start[-4:]) + 1))
    E = 0
    for i in ticker500:
        E_new = ExpectedAnnualReturn(i, start)
        if E_new > E:
            E = E_new
            s = i
                
    df1 = SP500[s].loc[(SP500.index >= ClosestDate(start)) & \
              (SP500.index <= ClosestDate(end))]
    df1.resample('D')
    stock = np.array((df1.pct_change() + 1).cumprod())

    return stock

tic = time.time()
df_M1 = pd.DataFrame()
i = 0
N = len(days)
df_M1 = pd.DataFrame()
for x in days:
    i += 1
    try:
        df_M1[x] = RandsomStockAnnualReturn()[:250]
    except:
        pass
    print(int(i / N * 100))
print(str(int(time.time() - tic)) + ' secs')
df_M1.to_csv(myPath + 'M1.csv')


df_M1 = pd.read_csv(myPath + 'M1.csv', index_col = [0])

data = df_M1
title = 'Daily Greatest Expected Return - Annual Return'
ylabel = 'Annual Return (%)'
plt.figure(figsize=[15, 10])
plt.plot(df_M1 - 1, linewidth = 1, c = 'b', alpha = 0.1)
plt.plot((df_M1 - 1).quantile([0.05, 0.25, 0.5, 0.75, 0.95], axis = 1).T, linewidth = 3, c = 'r')
plt.grid(axis = 'y')
plt.title('Daily Greatest Expected Return - Annual Return')
plt.xlabel('Time (One Year = 250 trading days)')
plt.ylabel('Annual Return (%)')
plt.show()

def RandomStockAnnualPremium():
        
    start = random.choice(days).strftime('%m/%d/%Y')
    end = ClosestDate(start[:6] + str(int(start[-4:]) + 1))
    E = 0
    for i in ticker500:
        E_new = ExpectedAnnualReturn(i, start)
        if E_new > E:
            E = E_new
            s = i                
    df1 = SP500[s].loc[(SP500.index >= ClosestDate(start)) & \
              (SP500.index <= ClosestDate(end))]
    df1.resample('D')
    stock = np.array((df1.pct_change() + 1).cumprod())
    r_f = (np.array([(1 + SP500['RiskFree'].loc[start]) ** (1/250)] * len(stock))).cumprod()
    riskPrem = stock - r_f
    
    return riskPrem

tic = time.time()
df_M2= pd.DataFrame()
i = 0
N = len(days)
df_M2 = pd.DataFrame()
for x in days:
    i += 1
    try:
        df_M2[x] = RandomStockAnnualPremium()[:250]
    except:
        pass
    print(int(i / N * 100))
print(str(int(time.time() - tic)) + ' secs')
df_M2.to_csv(myPath + 'M2.csv')


df_M2 = pd.read_csv(myPath + 'M2.csv', index_col = [0])
plt.figure(figsize=[15, 10])
plt.plot(df_M2, linewidth = 1, c = 'b', alpha = 0.1, label = 'Stocks')
plt.plot((df_M2).quantile([0.05, 0.25, 0.5, 0.75, 0.95], axis = 1).T, linewidth = 3, c = 'r', 
         label = 'Confidence Intervals')
plt.grid(axis = 'y')
plt.title('Daily Greatest Expected Return - Annual Risk Premium')
plt.xlabel('Time (One Year = 250 trading days)')
plt.ylabel('Annual Risk-Premium (%)')
plt.show()

####################################################################################################
### Buying Stock based on Expected Return only
####################################################################################################

def RandomSharpeAnnualReturn():
    
    start = random.choice(days).strftime('%m/%d/%Y')
    end = ClosestDate(start[:6] + str(int(start[-4:]) + 1))
    S = 0
    for i in ticker500:
        S_new = SharpeRatio(i, start)
        if S_new > S:
            S = S_new
            s = i                
    df1 = SP500[s].loc[(SP500.index >= ClosestDate(start)) & \
              (SP500.index <= ClosestDate(end))]
    df1.resample('D')
    stock = np.array((df1.pct_change() + 1).cumprod())

    return stock

tic = time.time()
df_M3 = pd.DataFrame()
i = 0
N = len(days)
df_M3 = pd.DataFrame()
for x in days:
    i += 1
    try:
        df_M3[x] = RandomSharpeAnnualReturn()[:250]
    except:
        pass
    print(int(i / N * 100))
print(str(int(time.time() - tic)) + ' secs')
df_M3.to_csv(myPath + 'M3.csv')


df_M3 = pd.read_csv(myPath + 'M3.csv', index_col = [0])
plt.figure(figsize=[15, 10])
plt.plot(df_M3 - 1, linewidth = 1, c = 'b', alpha = 0.1, label = 'Stocks')
plt.plot((df_M3 - 1).quantile([0.05, 0.25, 0.5, 0.75, 0.95], axis = 1).T, linewidth = 3, c = 'r', 
         label = 'Confidence Intervals')
plt.grid(axis = 'y')
plt.title('Daily Greatest Sharp-ratio - Annual Return')
plt.xlabel('Time (One Year = 250 trading days)')
plt.ylabel('Annual Return (%)')
plt.show()

def RandomSharpeAnnualPremium():
    
    start = random.choice(days).strftime('%m/%d/%Y')
    end = ClosestDate(start[:6] + str(int(start[-4:]) + 1))
    S = 0
    for i in ticker500:
        S_new = SharpeRatio(i, start)
        if S_new > S:
            S = S_new
            s = i                
    df1 = SP500[s].loc[(SP500.index >= ClosestDate(start)) & \
              (SP500.index <= ClosestDate(end))]
    df1.resample('D')
    stock = np.array((df1.pct_change() + 1).cumprod())
    r_f = (np.array([(1 + SP500['RiskFree'].loc[start]) ** (1/250)] * len(stock))).cumprod()
    riskPrem = stock - r_f

    return riskPrem

tic = time.time()
df_M4 = pd.DataFrame()
i = 0
N = len(days)
df_M4 = pd.DataFrame()
for x in days:
    i += 1
    try:
        df_M4[x] = RandomSharpeAnnualPremium()[:250]
    except:
        pass
    print(int(i / N * 100))
print(str(int(time.time() - tic)) + ' secs')
df_M4.to_csv(myPath + 'M4.csv')


df_M4 = pd.read_csv(myPath + 'M4.csv', index_col = [0])
plt.figure(figsize=[15, 10])
plt.plot(df_M4, linewidth = 1, c = 'b', alpha = 0.1, label = 'Stocks')
plt.plot((df_M4).quantile([0.05, 0.25, 0.5, 0.75, 0.95], axis = 1).T, linewidth = 3, c = 'r', 
         label = 'Confidence Intervals')
plt.grid(axis = 'y')
plt.title('Daily Greatest Sharp-ratio - Annual Risk Premium')
plt.xlabel('Time (One Year = 250 trading days)')
plt.ylabel('Annual Risk-Premium (%)')
plt.show()

####################################################################################################
### Buying a Stock based with greatest Beta only
####################################################################################################

def RandomBetaAnnualReturn():
    
    start = random.choice(days).strftime('%m/%d/%Y')
    end = ClosestDate(start[:6] + str(int(start[-4:]) + 1))
    B = 0
    for i in ticker500:
        B_new = BetaStock(i, start)
        if B_new > B:
            B = B_new
            s = i        
    df1 = SP500[s].loc[(SP500.index >= ClosestDate(start)) & \
              (SP500.index <= ClosestDate(end))]
    df1.resample('D')
    stock = np.array((df1.pct_change() + 1).cumprod())
    
    return stock

tic = time.time()
df_M5 = pd.DataFrame()
i = 0
N = len(days)
for x in days:
    i += 1
    try:
        df_M5[x] = RandomBetaAnnualReturn()[:250]
    except:
        pass
    print(int(i / N * 100))
print(str(int(time.time() - tic)) + ' secs')
df_M5.to_csv(myPath + 'M5.csv')

df_M5 = pd.read_csv(myPath + 'M5.csv', index_col = [0])
plt.figure(figsize=[15, 10])
plt.plot(df_M5 - 1, linewidth = 1, c = 'b', alpha = 0.1, label = 'Stocks')
plt.plot((df_M5 - 1).quantile([0.05, 0.25, 0.5, 0.75, 0.95], axis = 1).T, linewidth = 3, c = 'r', 
         label = 'Confidence Intervals')
plt.grid(axis = 'y')
plt.title('Daily Greatest Beta Return - Annual Return')
plt.xlabel('Time (One Year = 250 trading days)')
plt.ylabel('Annual Return (%)')
plt.show()

def RandomBetaAnnualPremium():
    
    start = random.choice(days).strftime('%m/%d/%Y')
    end = ClosestDate(start[:6] + str(int(start[-4:]) + 1))
    B = 0
    for i in ticker500:
        B_new = BetaStock(i, start)
        if B_new > B:
            B = B_new
            s = i    
    df1 = SP500[s].loc[(SP500.index >= ClosestDate(start)) & \
              (SP500.index <= ClosestDate(end))]
    df1.resample('D')
    stock = np.array((df1.pct_change() + 1).cumprod())
    r_f = (np.array([(1 + SP500['RiskFree'].loc[start]) ** (1/250)] * len(stock))).cumprod()
    prem = stock - r_f

    return prem


tic = time.time()
df_M6 = pd.DataFrame()
i = 0
N = len(days)
for x in days:
    i += 1
    try:
        df_M6[x] = RandomBetaAnnualPremium()[:250]
    except:
        pass
    print(int(i / N * 100))
print(str(int(time.time() - tic)) + ' secs')
df_M6.to_csv(myPath + 'M6.csv')


df_M6 = pd.read_csv(myPath + 'M6.csv', index_col = [0])
plt.figure(figsize=[15, 10])
plt.plot(df_M6 - 1, linewidth = 1, c = 'b', alpha = 0.1, label = 'Stocks')
plt.plot((df_M6 - 1).quantile([0.05, 0.25, 0.5, 0.75, 0.95], axis = 1).T, linewidth = 3, c = 'r', 
         label = 'Confidence Intervals')
plt.grid(axis = 'y')
plt.title('Daily Greatest Beta Return - Annual Risk Premium')
plt.xlabel('Time (One Year = 250 trading days)')
plt.ylabel('Annual Risk-Premium (%)')
plt.show()

####################################################################################################
### Buying a Stock based on CAPM only
####################################################################################################

def RandomCAPMAnnualReturn():
    
    start = random.choice(days).strftime('%m/%d/%Y')
    end = ClosestDate(start[:6] + str(int(start[-4:]) + 1))
    C = 0
    for i in ticker500:
        C_new = CAPMStock(i, start)
        if C_new > C:
            C = C_new
            s = i    
                
    df1 = SP500[s].loc[(SP500.index >= ClosestDate(start)) & \
              (SP500.index <= ClosestDate(end))]
    df1.resample('D')
    stock = np.array((df1.pct_change() + 1).cumprod())

    return stock

tic = time.time()
df_M7 = pd.DataFrame()
i = 0
N = len(days)
for x in days:
    i += 1
    try:
        df_M7[x] = RandomCAPMAnnualReturn()[:250]
    except:
        pass
    print(int(i / N * 100))
print(str(int(time.time() - tic)) + ' secs')
df_M7.to_csv(myPath + 'M7.csv')

df_M7 = pd.read_csv(myPath + 'M7.csv', index_col = [0])
plt.figure(figsize=[15, 10])
plt.plot(df_M7 - 1, linewidth = 1, c = 'b', alpha = 0.1, label = 'Stocks')
plt.plot((df_M7 - 1).quantile([0.05, 0.25, 0.5, 0.75, 0.95], axis = 1).T, linewidth = 3, c = 'r', 
         label = 'Confidence Intervals')
plt.grid(axis = 'y')
plt.title('Daily Greatest CAPM Return - Annual Return')
plt.xlabel('Time (One Year = 250 trading days)')
plt.ylabel('Annual Return (%)')
plt.show()

def RandomCAPMAnnualPremium():
    
    start = random.choice(days).strftime('%m/%d/%Y')
    end = ClosestDate(start[:6] + str(int(start[-4:]) + 1))
    C = 0
    for i in ticker500:
        C_new = CAPMStock(i, start)
        if C_new > C:
            C = C_new
            s = i                
    df1 = SP500[s].loc[(SP500.index >= ClosestDate(start)) & \
              (SP500.index <= ClosestDate(end))]
    df1.resample('D')
    stock = np.array((df1.pct_change() + 1).cumprod())
    r_f = (np.array([(1 + SP500['RiskFree'].loc[start]) ** (1/250)] * len(stock))).cumprod()
    prem = stock - r_f

    return prem


tic = time.time()
df_M8 = pd.DataFrame()
i = 0
N = len(days)
for x in days:
    i += 1
    try:
        df_M8[x] = RandomCAPMAnnualPremium()[:250]
    except:
        pass
    print(int(i / N * 100))
print(str(int(time.time() - tic)) + ' secs')
df_M8.to_csv(myPath + 'M8.csv')


df_M8 = pd.read_csv(myPath + 'M8.csv', index_col = [0])
plt.figure(figsize=[15, 10])
plt.plot(df_M8 - 1, linewidth = 1, c = 'b', alpha = 0.1, label = 'Stocks')
plt.plot((df_M8 - 1).quantile([0.05, 0.25, 0.5, 0.75, 0.95], axis = 1).T, linewidth = 3, c = 'r', 
         label = 'Confidence Intervals')
plt.grid(axis = 'y')
plt.title('Daily Greatest CAPM Return - Annual Risk Premium')
plt.xlabel('Time (One Year = 250 trading days)')
plt.ylabel('Annual Risk-Premium (%)')
plt.show()

####################################################################################################
### Buying a Stock based on Alpha only
####################################################################################################

def RandomAlphaAnnualReturn():
    
    start = random.choice(days).strftime('%m/%d/%Y')
    end = ClosestDate(start[:6] + str(int(start[-4:]) + 1))
    a = 0
    for i in ticker500:
        a_new = ExpectedAnnualReturn(i, start) - CAPMStock(i, start)
        if a_new > a:
            a = a_new
            s = i    
                
    df1 = SP500[s].loc[(SP500.index >= ClosestDate(start)) & \
              (SP500.index <= ClosestDate(end))]
    df1.resample('D')
    stock = np.array((df1.pct_change() + 1).cumprod())

    return stock


tic = time.time()
df_M9 = pd.DataFrame()
i = 0
N = len(days)
for x in days:
    i += 1
    try:
        df_M9[x] = RandomAlphaAnnualReturn()[:250]
    except:
        pass
    print(int(i / N * 100))
print(str(int(time.time() - tic)) + ' secs')
df_M9.to_csv(myPath + 'M9.csv')


df_M9 = pd.read_csv(myPath + 'M9.csv', index_col = [0])
plt.figure(figsize=[15, 10])
plt.plot(df_M9 - 1, linewidth = 1, c = 'b', alpha = 0.1, label = 'Stocks')
plt.plot((df_M9 - 1).quantile([0.05, 0.25, 0.5, 0.75, 0.95], axis = 1).T, linewidth = 3, c = 'r', 
         label = 'Confidence Intervals')
plt.grid(axis = 'y')
plt.title('Daily Greatest Alpha Return - Annual Return')
plt.xlabel('Time (One Year = 250 trading days)')
plt.ylabel('Annual Return (%)')
plt.show()

def RandomAlphaAnnualPremium():
    
    start = random.choice(days).strftime('%m/%d/%Y')
    end = ClosestDate(start[:6] + str(int(start[-4:]) + 1))
    a = 0
    for i in ticker500:
        a_new = ExpectedAnnualReturn(i, start) - CAPMStock(i, start)
        if a_new > a:
            a = a_new
            s = i    
                
    df1 = SP500[s].loc[(SP500.index >= ClosestDate(start)) & \
              (SP500.index <= ClosestDate(end))]
    df1.resample('D')
    stock = np.array((df1.pct_change() + 1).cumprod())
    r_f = (np.array([(1 + SP500['RiskFree'].loc[start]) ** (1/250)] * len(stock))).cumprod()
    prem = stock - r_f 

    return prem


tic = time.time()
df_M10 = pd.DataFrame()
i = 0
N = len(days)
for x in days:
    i += 1
    try:
        df_M10[x] = RandomAlphaAnnualPremium()[:250]
    except:
        pass
    print(int(i / N * 100))
print(str(int(time.time() - tic)) + ' secs')
df_M10.to_csv(myPath + 'M10.csv')


df_M10 = pd.read_csv(myPath + 'M10.csv', index_col = [0])
plt.figure(figsize=[15, 10])
plt.plot(df_M10 - 1, linewidth = 1, c = 'b', alpha = 0.1, label = 'Stocks')
plt.plot((df_M10 - 1).quantile([0.05, 0.25, 0.5, 0.75, 0.95], axis = 1).T, linewidth = 3, c = 'r', 
         label = 'Confidence Intervals')
plt.grid(axis = 'y')
plt.title('Daily Greatest Alpha Return - Annual Risk Premium')
plt.xlabel('Time (One Year = 250 trading days)')
plt.ylabel('Annual Risk-Premium (%)')
plt.show()

####################################################################################################
### Buying in January vs. December !!! This did not work very well at all, look into more!!!
####################################################################################################

# =============================================================================
# def RandomJanDecReturn(month = 'both'):    
#     
#     #Find dates    
#     s = random.choice(ticker500)
#     d1 = random.choice(range(1,32))
#     y1 = random.choice(range(2011, 2020))
#     dec = ClosestDate(str(y1) + '/' + str(12) + '/' + str(d1))
#     d2 = random.choice(range(1,32))
#     y2 = y1 + 1
#     jan = ClosestDate(str(y2) + '/' + str(1) + '/' + str(d2))
#     end = ClosestDate(str(y2 + 1) + '/' + str(10) + '/' + str(d2))
#     
#     if month in ['D', 'both']:
#         df_dec = pd.DataFrame()
#         df_dec[s] = SP500[s].loc[(SP500.index >= dec) & (SP500.index <= end)]
#         df_dec[s].resample('D')
#         stock_dec = np.array((df_dec[s].pct_change() + 1).cumprod())
#         stock_dec
#         if month == 'D':
#             return stock_dec
#     
#     if month in ['J', 'both']:
#         df_jan = pd.DataFrame()
#         df_jan[s] = SP500[s].loc[(SP500.index >= jan) & (SP500.index <= end)]
#         df_jan[s].resample('D')
#         stock_jan = np.array((df_jan[s].pct_change() + 1).cumprod())
#         if month == 'J':
#             return stock_jan
#     
#     if month == 'both':
#         df = df_dec.merge(df_jan,
#                       left_index = True,
#                       right_index = True,
#                       how = 'left')
#         df = (df.pct_change() + 1).cumprod()
#         return df
# 
# tic = time.time()
# df_M9Dec = pd.DataFrame()
# df_M9Jan = pd.DataFrame()
# i = 0
# N = len(days)
# for x in days:
#     i += 1
#     try:
#         df_M9Dec[x] = RandomJanDecReturn('D')[:375]
#         df_M9Jan[x] = RandomJanDecReturn('J')[:375]
#     except:
#         pass
#     print(int(i / N * 100))
# print(str(int(time.time() - tic)) + ' secs')
# df_M9Dec.to_csv(myPath + 'M9Dec.csv')
# df_M9Jan.to_csv(myPath + 'M9Jan.csv')
# 
# df_M9Dec = pd.read_csv(myPath + 'M9Dec.csv', index_col = [0])
# df_M9Jan = pd.read_csv(myPath + 'M9Jan.csv', index_col = [0])
# plt.figure(figsize=[20, 15])
# plt.plot(df_M9Dec - 1, linewidth = 1, c = 'b', alpha = 0.1, label = 'December')
# plt.plot(df_M9Jan - 1, linewidth = 1, c = 'r', alpha = 0.1, label = 'January')
# plt.grid(axis = 'y')
# plt.title('Daily Greatest Alpha Return - Annual Risk Premium')
# plt.xlabel('Time (One Year = 250 trading days)')
# plt.ylabel('Annual Risk-Premium (%)')
# plt.show()
# =============================================================================

####################################################################################################
### Cumulative Distribution Function Returns
####################################################################################################
df_BaseReturn = pd.read_csv(myPath + 'BaseReturn.csv', index_col = [0])
df_MktReturn = pd.read_csv(myPath + 'MarketReturn.csv', index_col = [0])
df_M1 = pd.read_csv(myPath + 'M1.csv', index_col = [0])
df_M3 = pd.read_csv(myPath + 'M3.csv', index_col = [0])
df_M5 = pd.read_csv(myPath + 'M5.csv', index_col = [0])
df_M7 = pd.read_csv(myPath + 'M7.csv', index_col = [0])
df_M9 = pd.read_csv(myPath + 'M9.csv', index_col = [0])

plt.figure(figsize=[15, 10])

dfBR = df_BaseReturn.iloc[-1].dropna()[:500]
valuesBR, baseBR = np.histogram(dfBR, bins=100)
cumulativeBR = np.cumsum(valuesBR)
plt.plot(baseBR[:-1], cumulativeBR, label = "Baseline Return")

dfMR = df_MktReturn.iloc[-1].dropna()[:500]
valuesMR, baseMR = np.histogram(dfMR, bins=100)
cumulativeMR = np.cumsum(valuesMR)
plt.plot(baseMR[:-1], cumulativeMR, label = "Market Return")

dfM1 = df_M1.iloc[-1].dropna()[:500]
values1, base1 = np.histogram(dfM1, bins=100)
cumulative1 = np.cumsum(values1)
plt.plot(base1[:-1], cumulative1, label = "Expected Return")

dfM3 = df_M3.iloc[-1].dropna()[:500]
values3, base3 = np.histogram(dfM3, bins=100)
cumulative3 = np.cumsum(values3)
plt.plot(base3[:-1], cumulative3, label = "Sharpe-ratio")

dfM5 = df_M5.iloc[-1].dropna()[:500]
values5, base5 = np.histogram(dfM5, bins=100)
cumulative5 = np.cumsum(values5)
plt.plot(base5[:-1], cumulative5, label = "Beta")

dfM7 = df_M7.iloc[-1].dropna()[:500]
values7, base7 = np.histogram(dfM7, bins=100)
cumulative7 = np.cumsum(values7)
plt.plot(base7[:-1], cumulative7, label = "CAPM")

dfM9 = df_M9.iloc[-1].dropna()[:500]
values9, base9 = np.histogram(dfM9, bins=100)
cumulative9 = np.cumsum(values9)
plt.plot(base9[:-1], cumulative9, label = "Alpha")

datarange = dfM1.append(dfM3).append(dfM5).append(dfM7).append(dfM9).append(dfBR).append(dfMR)

plt.xticks(np.arange(int(min(datarange) / 0.25) * 0.25,
                     int(max(datarange / 0.25 + 2) * 0.25),
                         0.25))
plt.yticks(range(0, len(days), int(len(days) / 11)), range(0, 110, 10))
plt.title('Cumulative Distribution Functions (Annual Returns)')
plt.xlabel('Annual Return')
plt.ylabel('Probabilities')
plt.grid()
plt.legend()
plt.show()





####################################################################################################
### Cumulative Distribution Function Risk Premiums
####################################################################################################
df_BasePremium = pd.read_csv(myPath + 'BasePremium.csv', index_col = [0])
df_MktPremium = pd.read_csv(myPath + 'MarketPremium.csv', index_col = [0])
df_M2 = pd.read_csv(myPath + 'M2.csv', index_col = [0])
df_M4 = pd.read_csv(myPath + 'M4.csv', index_col = [0])
df_M6 = pd.read_csv(myPath + 'M6.csv', index_col = [0])
df_M8 = pd.read_csv(myPath + 'M8.csv', index_col = [0])
df_M10 = pd.read_csv(myPath + 'M10.csv', index_col = [0])

plt.figure(figsize=[15, 10])

dfBP = df_BasePremium.iloc[-1].dropna()[:500]
valuesBP, baseBP = np.histogram(dfBP, bins=100)
cumulativeBP = np.cumsum(valuesBP)
plt.plot(baseBP[:-1], cumulativeBP, label = "Baseline Return")

dfMP = df_MktPremium.iloc[-1].dropna()[:500]
valuesMP, baseMP = np.histogram(dfMP, bins=100)
cumulativeMP = np.cumsum(valuesMP)
plt.plot(baseMP[:-1], cumulativeMP, label = "Market Return")

dfM2 = df_M2.iloc[-1].dropna()[:500]
values2, base2 = np.histogram(dfM2, bins=100)
cumulative2 = np.cumsum(values2)
plt.plot(base2[:-1], cumulative2, label = "Expected Return")

dfM4 = df_M4.iloc[-1].dropna()[:500]
values4, base4 = np.histogram(dfM4, bins=100)
cumulative4 = np.cumsum(values4)
plt.plot(base4[:-1], cumulative4, label = "Sharpe-ratio")

dfM6 = df_M6.iloc[-1].dropna()[:500]
values6, base6 = np.histogram(dfM6, bins=100)
cumulative6 = np.cumsum(values6)
plt.plot(base6[:-1], cumulative6, label = "Beta")

dfM8 = df_M8.iloc[-1].dropna()[:500]
values8, base8 = np.histogram(dfM8, bins=100)
cumulative8 = np.cumsum(values8)
plt.plot(base8[:-1], cumulative8, label = "CAPM")

dfM10 = df_M10.iloc[-1].dropna()[:500]
values10, base10 = np.histogram(dfM10, bins=100)
cumulative10 = np.cumsum(values10)
plt.plot(base10[:-1], cumulative10, label = "Alpha")

datarange = dfM2.append(dfM4).append(dfM6).append(dfM8).append(dfM10).append(dfBP).append(dfMP)

plt.xticks(np.arange(int(min(datarange) / 0.25) * 0.25,
                     int(max(datarange / 0.25 + 2) * 0.25),
                         0.25))
plt.yticks(range(0, len(days), int(len(days) / 11)), range(0, 110, 10))
plt.title('Cumulative Distribution Functions (Annual Risk Premiums)')
plt.xlabel('Annual Return')
plt.ylabel('Probabilities')
plt.grid()
plt.legend()
plt.show()

####################################################################################################
### 
####################################################################################################


####################################################################################################
### 
####################################################################################################

####################################################################################################
### 
####################################################################################################

####################################################################################################
### Use Sharpe Ratio to find top stocks by year
####################################################################################################

# =============================================================================
# #Create empty dataframe with Date index
# JoesPort = pd.DataFrame(columns = ['Date', 'Ticker', 'Qty'])
# JoesPort = JoesPort.set_index(['Date'])
# 
# for y in list(range(2011, 2015, 1)):
#     
#     calc_date = ClosestDate(str(y) + '0101')
#     print(calc_date)
#     sharpe = 0
#     df = SP500.loc[SP500.index <= calc_date].iloc[:,:-2]
#     
#     for i in set(df.columns):
#         if StockCost(i, calc_date,) <= 100:
#             s = SharpeRatio(i, calc_date)
#             if s > sharpe:
#                 sharpe = s
#                 t = i
#     data = {'Ticker':t, 'Qty':1}
#     JoesPort = JoesPort.append(pd.DataFrame(data, index = [calc_date]))
#     
#     print(JoesPort)
# =============================================================================


####################################################################################################
### 
####################################################################################################
#Create empty dataframe with Date index
# =============================================================================
# JoesPort = pd.DataFrame(columns = ['Date', 'Ticker', 'Qty'])
# JoesPort = JoesPort.set_index(['Date'])
# 
# for y in list(range(2011, 2015, 1)):
#     
#     calc_date = ClosestDate(str(y) + '0101')
#     print(calc_date)
#     sharpe = 0
#     df = SP500.loc[SP500.index <= calc_date].iloc[:,:-2]
#     
#     for i in set(df.columns):
#         if StockCost(i, calc_date,) <= 100:
#             s = SharpeRatio(i, calc_date)
#             if s > sharpe:
#                 sharpe = s
#                 t = i
#     data = {'Ticker':t, 'Qty':1}
#     JoesPort = JoesPort.append(pd.DataFrame(data, index = [calc_date]))
#     
#     print(JoesPort)
# =============================================================================
####################################################################################################
### Beta of a Stock with respect to the Market (Measure of systematic risk)
#       Beta = Cov[StockReturn, MarketReturn] over the Variance of the Market
####################################################################################################


##################################################################################################
### Expected Return of a Portfolio
####################################################################################################

# =============================================================================
# def ExpectedPortReturn(portfolio, date):
#     '''Returns the Expected Return of a Portfolio'''
#     
#     P = portfolio
#     d = ClosestDate(date)
#     
#     Total = 0
#     for index, row in P.iterrows():
#         stock = row['Ticker']
#         qty = row['Qty']
#         price = StockCost(stock, d, qty)
#         Total = Total + price
#     
#     E = 0
#     for index, row in P.iterrows():
#         stock = row['Ticker']
#         qty = row['Qty']
#         price = ExpectedAnnualReturn(stock, d)
#         weight = price / Total
#         E = E + (weight * price)
#     
#     return E
# 
# ExpectedPortReturn(JoesPort, '2020-01-01')
# =============================================================================

####################################################################################################
### Plot a portfolio value in time-series as stocks are bought and sold
####################################################################################################

# =============================================================================
# def PortLifePlot(portfolio):
#     '''Plots the lifetime of a portfolio as stocs are bought and sold'''    
# 
#     P = portfolio    
#     df = pd.DataFrame(index = SP500.index[SP500.index >= P.index[0]])
#     df_mkt = df.merge(SP500['Market'],
#                       left_index = True,
#                       right_index = True,
#                       how = 'left')
#     for index, row in P.iterrows():
#         symbol = row['Ticker']
#         qty = row['Qty']
#         date = index        
#         if symbol in set(df.columns):
#             df1 = df[symbol]
#             df2 = SP500[symbol].loc[SP500.index >= date] * qty
#             df1 = df.merge(df2,
#                            left_index = True,
#                            right_index = True,
#                            how = 'left')
#             df[symbol] = df1.sum(axis = 1)        
#         else:
#             df = df.merge(SP500[symbol].loc[SP500.index >= date] * qty,
#                           left_index = True,
#                           right_index = True,
#                           how = 'outer')    
#     df['portfolio'] = df.sum(axis = 1)
#     df['Market'] = df_mkt
#     df.plot()
#     
#     return 
# 
# PortLifePlot(JoesPort)
# =============================================================================

####################################################################################################
### Plot a portfolio in time-series as stocks are bought and sold
####################################################################################################

# =============================================================================
# def PortLifeReturnPlot(portfolio):
#     '''Plots the lifetime of a portfolio as stocs are bought and sold'''    
# 
#     P = portfolio    
#     df = pd.DataFrame(index = SP500.index[SP500.index >= P.index[0]])
#     df_mkt = df.merge(SP500['Market'],
#                   left_index = True,
#                   right_index = True,
#                   how = 'left')
#     for index, row in P.iterrows():
#         symbol = row['Ticker']
#         qty = row['Qty']
#         date = index        
#         if symbol in set(df.columns):
#             df1 = df[symbol]
#             df2 = SP500[symbol].loc[SP500.index >= date] * qty
#             df1 = df.merge(df2,
#                            left_index = True,
#                            right_index = True,
#                            how = 'left')
#             df[symbol] = df1.sum(axis = 1)        
#         else:
#             df = df.merge(SP500[symbol].loc[SP500.index >= date] * qty,
#                           left_index = True,
#                           right_index = True,
#                           how = 'outer')    
#     df['portfolio'] = df.sum(axis = 1)
#     df['Market'] = df_mkt
#     df_ret = (df / df.iloc[0]).fillna(method = 'backfill')
#     df_ret.plot()
#     
#     return 
# 
# PortLifeReturnPlot(JoesPort)
# =============================================================================
