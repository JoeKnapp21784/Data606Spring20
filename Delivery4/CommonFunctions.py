'''Joseph Knapp - Data606Capstone

File Info: This file holds user defined functions which are commonly used throughout project'''


####################################################################################################
### Libraries 
####################################################################################################

import pandas as pd

####################################################################################################
### Main Path
####################################################################################################

myPath = myPath = 'C:\\Data606\\'

####################################################################################################
### Downloads up to date ticker symbols that are in S&P500 from Wikipedia site:
#       https://en.wikipedia.org/wiki/List_of_S%26P_500_companies
####################################################################################################

def SP500TickerData():
    '''Downloads the S&P500 ticker symbols intoo a list'''
    
    table=pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    df = table[0]
    df.drop(['SEC filings', 'Headquarters Location', 'CIK', 'Founded', 'Date first added'], 
            axis = 1, 
            inplace = True)
    df = df.sort_values(by = ['Symbol']).reset_index(drop = True)
    df.columns = ['Symbol', 'Security', 'Sector', 'SubIndustry']
    droplist = set(['BF.B', 'BRK.B', 'TT'])   #these throw errors 
    for i in droplist:
        df = df[df.Symbol != i]
    df.reset_index(drop = True, inplace = True)
    
    return df

####################################################################################################
### Downloads up to date ticker symbols that are in S&P500 from Wikipedia site:
#       https://en.wikipedia.org/wiki/List_of_S%26P_500_companies
####################################################################################################

def SP400TickerData():
    '''Downloads the S&P500 ticker symbols intoo a list'''
    
    table=pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_400_companies')
    df = table[0]
    df.columns
    df.drop(['SEC filings'], axis = 1, inplace = True)
    df.columns = ['Security', 'Symbol', 'Sector', 'SubIndustry']
    df = df.sort_values(by = ['Symbol']).reset_index(drop = True)
    df = df.reindex(['Symbol', 'Security', 'Sector', 'SubIndustry'], axis = 1)
    
    return df

####################################################################################################
### Reads in clean dataframe and returns a time-series dataframes
####################################################################################################

def FormatData(FileName):
    '''This will take the Date column and convert it to a date type and make
    it the index'''
    
    df = pd.read_csv(FileName)
    df.Date = pd.to_datetime(df.Date)    #converts Date column to Date object
    df.set_index('Date', inplace = True) 

    return df
####################################################################################################
### Prints Progress
####################################################################################################

def PercentDone(i, N):
    print(str(round((i / N) * 100, len(str(N)) - 1)) + '%')
    return

































