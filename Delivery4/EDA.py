'''Joseph Knapp - Data606Capstone

File Info: This file holds different statistical models tried on data'''

####################################################################################################
### Libraries 
####################################################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

####################################################################################################
### Variables
####################################################################################################

myPath = 'C:\\Data606\\CleanData\\'

####################################################################################################
### Plot the CDF of the different Metrics given their return percentiles
####################################################################################################
#Data and variables
df_mod = pd.read_csv(myPath + 'ModelingData.csv', index_col = [0])
q = 0.90
par00 = ['ExpectedMarketReturn', 'ExpectedMarketReturnPr1Yr', \
         'ExpectedReturn', 'ExpectedReturnPr1Yr', \
         'Volatility', 'VolatilityPr1Yr', \
         'Sharpe', 'SharpePr1Yr', \
         'Beta', 'BetaPr1Yr', \
         'CAPM', 'CAPMPr1Yr', \
         'Alpha', 'AlphaPr1Yr', \
         'StockPrice']

#Annual Returns
plt.figure(figsize = [15,10])
for m in par00:
    df = df_mod[df_mod[m] > df_mod[m].quantile(q)]
    x = np.sort(df['AnnualReturn'])
    y = np.arange(1, len(x) + 1) / len(x)
    plt.plot(x, y, label = m)
plt.legend()
plt.grid(axis = 'both')
plt.xlabel('Annual Return')
plt.ylabel('Probabilities')
plt.title('CDF of Metrics and their Annual Returns')
plt.show()

#Annual Risk Premium
plt.figure(figsize = [15,10])
for m in par00:
    df = df_mod[df_mod[m] > df_mod[m].quantile(q)]
    x = np.sort(df['AnnualRiskPremium'])
    y = np.arange(1, len(x) + 1) / len(x)
    plt.plot(x, y, label = m)
plt.legend()
plt.grid(axis = 'both')
plt.show()


#Annual Market Premium
plt.figure(figsize = [15,10])
for m in par00:
    df = df_mod[df_mod[m] > df_mod[m].quantile(q)]
    x = np.sort(df['AnnualMarketPremium'])
    y = np.arange(1, len(x) + 1) / len(x)
    plt.plot(x, y, label = m)
plt.legend()
plt.grid(axis = 'both')
plt.show()

####################################################################################################
### Visualize the Metrics
####################################################################################################
#Box Plots
df_mod = pd.read_csv(myPath + 'ModelingData.csv', index_col = [0])
df_mod.columns
df_mod01 = df_mod[['ExpectedMarketReturn', 'ExpectedMarketReturnPr1Yr',
                   'ExpectedReturn', 'ExpectedReturnPr1Yr',
                   'Volatility', 'VolatilityPr1Yr',
                   'Sharpe', 'SharpePr1Yr',
                   'Beta', 'BetaPr1Yr',
                   'CAPM', 'CAPMPr1Yr',
                   'Alpha', 'AlphaPr1Yr',
                   'TVarLow', 'TVarLowPr1Yr',
                   'TVarHigh', 'TVarHighPr1Yr',
                   'AnnualReturn', 'AnnualRiskPremium', 'AnnualMarketPremium']]

df = df_mod01[['ExpectedMarketReturn', 'ExpectedMarketReturnPr1Yr',
                   'ExpectedReturn', 'ExpectedReturnPr1Yr']]
df = df[df < 2]
for column in df:
    plt.figure()
    df.plot(kind = 'box')
plt.title('Boxplots of Annual Expected Market and Stock Returns')
plt.ylabel('Annualized Returns')
plt.xticks(rotation = 45)
plt.show()

df = df_mod01[['Volatility', 'VolatilityPr1Yr']]
df = df[df < 2]
for column in df:
    plt.figure()
    df.plot(kind = 'box')
plt.title('Boxplots of Annual Volatility')
plt.ylabel('Volatility of Annual Returns')
plt.xticks(rotation = 45)
plt.show()

df = df_mod01[['Sharpe', 'SharpePr1Yr']]
df = df[df < 2]
for column in df:
    plt.figure()
    df.plot(kind = 'box')
plt.title('Boxplots of Annual Sharpe ratio')
plt.ylabel('Sharpe ratio')
plt.xticks(rotation = 45)
plt.show()

df = df_mod01[['Beta', 'BetaPr1Yr']]
df = df[df < 2]
for column in df:
    plt.figure()
    df.plot(kind = 'box')
plt.title('Boxplots of Annual Beta')
plt.ylabel('Beta')
plt.xticks(rotation = 45)
plt.show()



#Scatterplots
df_mod = pd.read_csv(myPath + 'ModelingData.csv', index_col = [0])
df_mod_col = list(df_mod.columns.drop(['Date', 'Symbol', 'AnnualReturn', 'AnnualRiskPremium', \
                                       'AnnualMarketPremium', 'Sector', 'SubIndustry']))
dep_var = 'AnnualReturn' #Choose dependent variable
fig, axs = plt.subplots(4, 5, sharey = True, figsize = [10, 6])
i = 0
for ax in axs.flatten():
    ax.scatter(df_mod[df_mod_col[i]], df_mod[dep_var],
               alpha = 0.2,
               s = 10)
    ax.set_xlabel(df_mod_col[i])
    ax.set_ylabel(dep_var)
    i += 1


 
#Heatmap of correlations
df_mod = pd.read_csv(myPath + 'ModelingData.csv', index_col = [0])
df_mod_col = list(df_mod.columns.drop(['Symbol', 'Date', 'Sector', 'SubIndustry']))

fig, ax = plt.subplots(figsize = [15,10])
corr_data = df_mod[df_mod_col].corr()
ax = sns.heatmap(corr_data, 
                 vmin = -1, vmax = 1,
                 center = 0, 
                 annot = True)

# =============================================================================
# #Distribution of each variable
# df_mod = pd.read_csv(myPath + 'ModelingData.csv', index_col = [0])
# df_mod_col = list(df_mod.columns.drop(['Symbol', 'Date', 'Sector', 'SubIndustry']))
# fig, axs = plt.subplots(6, 3, figsize = [15,20])
# i = 0
# for ax in axs.flatten():
#     ax = sns.distplot(df_mod[df_mod_col[i]], hist = False, label = 'All')
#     ax = sns.distplot(df_mod[df_mod[dep_var] > 0][df_mod_col[i]], hist = False, label = 'Top')
#     ax = sns.distplot(df_mod[df_mod[dep_var] < 0][df_mod_col[i]], hist = False, label = 'Bot')
#     i += 1
# =============================================================================













