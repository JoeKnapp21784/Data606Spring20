#Libraries used
import pandas as pd
import matplotlib.pyplot as plt

############################################
### Enter path to files below...............................................
############################################
myPath = '''C:\\Data606\\CleanData\\'''

############################################
### Create a function that reads in data and correctly formats
############################################
def FormatData(FileName):
    '''This will take the Date column and convert it to a date type and make
    it the index'''
    
    df = pd.read_csv(myPath + FileName)
    df.Date = pd.to_datetime(df.Date)    #converts Date column to Date object
    df.set_index('Date', inplace = True) 

    return df

############################################
### Create a function that turns daily returns into overal returns
############################################
def DollarReturn(Symbol):  
    
    r = [1]
    daily_rates = SPRates[Symbol] + 1
    x = 1
    for i in daily_rates[1:]:
        x = x * i
        r.append(x)
    
    return r  

############################################
### Calculating Beta
############################################
###Read in S&P500 Index data
SPIndex = FormatData('SP500Overall.csv')['Rate'].to_frame()

###Read in S&P500 Individual Stock data
SPRates = FormatData('SP500StockReturns.csv')

###Beta function
def BetaMarket(StockSymbol):


    Covariance = SPRates[StockSymbol].cov(SPRates['Market'])
    MarketVariance = SPRates['Market'].var()
    Beta = Covariance / MarketVariance
    
    return Beta

###Find Beta of each Stock
Symbols = SPRates.columns
BetaDict = {}
for i in Symbols:
    BetaDict[i] = BetaMarket(i)
    print(i, BetaMarket(i))

Beta_df = pd.DataFrame.from_dict(BetaDict, orient = 'index')
Beta_df.columns = ['Beta']

###Find companies large Betas
Beta_df.nlargest(5, 'Beta').index[2:]
Beta_df.nsmallest(5, 'Beta').index[2:]

###Plot prices vs market and see how well beta does
for i in Beta_df.nlargest(4, 'Beta').index[1:]:
    plt.plot(SPRates.index, DollarReturn(i), label = i)
plt.plot(SPRates.index, DollarReturn('Market'), label = 'Market')
plt.legend()
plt.show()

for i in Beta_df.nsmallest(4, 'Beta').index[1:]:
    plt.plot(SPRates.index, DollarReturn(i), label = i)
plt.plot(SPRates.index, DollarReturn('Market'), label = 'Market')
plt.legend()
plt.show()




