#Libraries used
import pandas as pd
import os

############################################
### Enter path to files below...............................................
############################################
myPath = '''C:\\Data606\\'''

############################################
###TBill Data from FRED 10 TBill rate
############################################
#Read in file
TBills = pd.read_csv(myPath + 'OriginalData\\FRED10YrTreasurityRate.csv',
                     header = 0,
                     names = ['Date', 'Rate'])

###Get dataframe info
TBills.info()    #need to fix Date column datatype

###Make Data column in datatype Date and make it the index
TBills.Date.dtypes   #date column is an object type
TBills.Date = pd.to_datetime(TBills.Date)    #converts Date column to Date object
TBills.set_index('Date', inplace = True)    #set the Date as the index

###View head and tail of dataframe
TBills.head(10)
TBills.tail(10)    #Whats up with the periods????????

###View dates of '.'
TBills[TBills.Rate == '.']   #These are holidays, I will remove them
TBills = TBills[TBills.Rate != '.']    #Removed all the rows with '.'

###Convert Rate to a float type and then to decimals
TBills.Rate.dtypes
TBills.Rate = TBills.Rate.astype(float)
TBills.Rate = TBills.Rate.apply(lambda x: x / 100)

###View it as a time series graph
TBills.plot()

###Output TBills to .csv
TBills.to_csv(myPath + 'CleanData\\TBillRate.csv')

############################################
###S&P500 Overall Index Data
############################################
###Read in file
SP500 = pd.read_csv(myPath + 'OriginalData\\YahooS&PHistoricalData.csv')

###Get dataframe info
SP500.info()

###Make Data column in datatype Date and make it the index
SP500.Date.dtypes   #date column is an object type
SP500.Date = pd.to_datetime(SP500.Date)    #converts Date column to Date object
SP500.set_index('Date', inplace = True)    #set the Date as the index

###Remove unecessary columns
SP500.columns
SP500.drop(['High', 'Low', 'Close', 'Adj Close'], axis = 1, inplace = True)

###plot Open and Volume with respect to time
SP500['Open'].plot()
SP500['Volume'].plot()   #hard to visualize, ill try smoothing
SP500['Volume'].rolling(window = 100).mean().plot()   #smoothed with a rolling average

###Output SP500 to .csv
SP500.to_csv(myPath + 'CleanData\\SP500Overall.csv')

############################################
### S&P500 Daily Stock Prices
############################################
###Read in file
prices = pd.read_csv(myPath + 'OriginalData\\S&P500DailyPrices.csv')

###Get dataframe info
prices.info()

###Remove unecessary columns
prices.columns    #check column names
prices.drop(['close', 'low', 'high', 'volume'], axis = 1, inplace = True)   #remove columns
prices.columns = ['Date', 'Symbol', 'Open']   #rename columns

###Need to pivot the dataframe and make the date the index and the columns for each stock
ticker = prices.Symbol.unique()   #get each symbol
len(ticker)   #501
prices.Date = pd.to_datetime(prices.Date)

###Pull date and price for each symbol out of prices dataframe
def StockPrices(Symbol):
    
    i = Symbol
    df = prices[prices['Symbol'] == i]
    df.set_index('Date', inplace = True)
    df.drop('Symbol', axis = 1, inplace = True)
    df.columns = [i]
    
    return df

###Merge all the data based on Date index
SPrice = pd.DataFrame()
for i in ticker:
    
    if SPrice.empty == True:
        SPrice = StockPrices(i)
    else:
        df_new = StockPrices(i)
        SPrice = SPrice.merge(df_new, left_index = True, right_index = True, how = 'outer')
    
    print(i)
    
###Output SPrice to .csv
SPrice.to_csv(myPath + 'CleanData\\SP500StockPrices.csv')








































############################################
### Combining aLL US Stock Daily Data
############################################
###Get Date and Price out of each .txt file
def GetStockData(FileName): 
    """Returns a DataFrame where the Date is the index and the column has the 
    opening price of the stock that day"""
    
    try:   #some of the files are empty and throw an error

        df = pd.read_csv(myPath + 'Stocks\\' + FileName)
        
        #make the date the index
        df.Date = pd.to_datetime(df.Date)      
        df.set_index('Date', inplace = True)
        
        #remove uneeded data
        df.drop(['High', 'Low', 'Close', 'OpenInt', 'Volume'], axis = 1, inplace = True)  
        
        #get stock symbol from file name and make it the column name
        symbol = FileName.split('.', 1)[0]
        df.columns = [symbol]

    except:
        df = pd.DataFrame()    #empty files get an empty dataframe
        
    return df

###Merge all the stock different stocks into one large dataframe
fileList = os.listdir(myPath + 'Stocks')
df = pd.DataFrame()
for i in fileList:
       
    if df.empty == True:
        df = GetStockData(i)
    else:
        df_new = GetStockData(i)
        df = df.merge(df_new, left_index = True, right_index = True, how = 'outer')
    print(i)

print(df)

###Make all Nan equal to zero
dfa = df
x = isnan(dfa)
dfa[x] = 0
    
############################################
### Combining aLL US Stock Daily Data
############################################

   






