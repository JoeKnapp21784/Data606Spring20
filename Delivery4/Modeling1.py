'''Joseph Knapp - Data606Capstone

File Info: This file holds different statistical models tried on data'''

####################################################################################################
### Libraries 
####################################################################################################

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.feature_selection import f_regression
from sklearn.metrics import confusion_matrix
import statsmodels.api as sm

####################################################################################################
### Variables
####################################################################################################

myPath = 'C:\\Data606\\CleanData\\'

####################################################################################################
### Read in Datasets
####################################################################################################
#Read in dataset
df_data = pd.read_csv(myPath + 'ModelingData.csv', index_col = [0])

#Dependent Variable list
Y_vars = ['AnnualReturn', 'AnnualRiskPremium', 'AnnualMarketPremium']

#Dataset with all metrics (historitcal and prior 1yr)
df_mod01 = pd.get_dummies(df_data.drop(['Symbol', 'Date', 'SubIndustry',
                                        'AnnualRiskPremium', 'AnnualMarketPremium'], 
                                       axis = 1),
                            drop_first = True)
df_data.columns

#Dataset with metrics that use all histrocial data
df_mod01Hist = pd.get_dummies(df_data[['StockPrice',
                                       'ExpectedMarketReturn',
                                       'ExpectedReturn',  
                                       'Volatility', 
                                       'Sharpe', 
                                       'Beta',
                                       'CAPM' ,
                                       'Alpha',
                                       'TVarLow',
                                       'TVarHigh',                        
                                       'Sector']],
                            drop_first = True)

#Dataset with metrics that use all past yrs worth of data
df_mod01Pr1Yr = pd.get_dummies(df_data[['ExpectedMarketReturnPr1Yr',
                                        'ExpectedReturnPr1Yr',
                                        'VolatilityPr1Yr',
                                        'SharpePr1Yr',
                                        'BetaPr1Yr',
                                        'CAPMPr1Yr',
                                        'AlphaPr1Yr',
                                        'TVarLowPr1Yr',
                                        'TVarHighPr1Yr',
                                        'Sector']],
                            drop_first = True)

# =============================================================================
# #Regular model
# X_train, X_test, Y_train, Y_test = train_test_split(df_mod01.drop('AnnualReturn', axis = 1), df_data[Y_vars[0]])
# X_train = sm.add_constant(X_train)
# reg01 = sm.OLS(Y_train, X_train).fit()
# reg01.summary()
# 
# 
# #Historical Data
# X_train, X_test, Y_train, Y_test = train_test_split(df_mod01Hist, df_data[Y_vars[0]])
# X_train = sm.add_constant(X_train)
# reg01 = sm.OLS(Y_train, X_train).fit()
# reg01.summary()
# 
# 
# #Pr1Yr Data
# X_train, X_test, Y_train, Y_test = train_test_split(df_mod01Pr1Yr, df_data[Y_vars[0]])
# X_train = sm.add_constant(X_train)
# X_train = sm.add_constant(X_train)
# reg01 = sm.OLS(Y_train, X_train).fit()
# reg01.summary()
# =============================================================================

####################################################################################################
### OLS Linear Regression using Forward Selection w/ K-Fold CV
####################################################################################################
# =============================================================================
# df_data = pd.read_csv(myPath + 'ModelingData.csv', index_col = [0])
# df_mod01 = pd.get_dummies(df_data.drop(['Symbol', 'Date', 'SubIndustry',
#                                         'AnnualRiskPremium', 'AnnualMarketPremium'], 
#                                        axis = 1), drop_first = True)
# col01 = ['NewParameter', 'Parameters', 'ParamQty', 'RSS', 'R-squared']
# df_params = pd.DataFrame(columns = col01)
# X_vars = list(df_mod01.columns.drop('AnnualReturn'))
# x_vars = []
# RSS = math.inf
# i = 0
# df_mod01 = df_mod01.sample(frac = 1)
# KFolds = 5
# N = len(X_vars)
# while len(X_vars) != 0:
#     start = len(X_vars)
#     var = ''
#     if i >= start:
#         RSS = math.inf
#         Score = 0
#     for x in X_vars:
#         k = 0
#         Score_list = []
#         RSS_list = []
#         while k < KFolds:
#             X_test = df_mod01.iloc[k * int(len(df_mod01)/KFolds):int(len(df_mod01)/KFolds) * (k + 1)]
#             X_train = df_mod01.drop(X_test.index)
#             X_train_data = X_train[[x] + x_vars]
#             X_test_data = X_test[[x] + x_vars]
#             Y_test = X_test['AnnualReturn']
#             Y_train = X_train['AnnualReturn']
#             reg = LinearRegression().fit(X_train_data,Y_train)
#             y_pred = reg.predict(X_test_data)
#             RSS_list.append(round(np.sum((Y_test - y_pred)**2),2))            
#             Score_list.append(round(reg.score(X_test_data, Y_test),4))
#             k += 1
#         RSS_new = np.mean(RSS_list)
#         Score = np.mean(Score_list)      
#         if RSS_new < RSS:
#             RSS = RSS_new
#             var = x
#             RSS_list_main = RSS_list
#             Score_list_main = Score_list
#     if var != '':
#         x_vars.append(var)
#         X_vars.remove(var)
#         print('--------')
#         print('Parameters: ' + str(x_vars))
#         print('RSS: ' + str(round(RSS,4)) + ' : ' + str(RSS_list_main))
#         print('R^2: ' + str(round(Score, 4)) + ' : ' + str(Score_list_main))
#         row1 = pd.Series([var, x_vars, len(x_vars),round(RSS,4), round(Score, 4)], index = col01)
#         df_params = df_params.append(row1, ignore_index = True)        
#     i += 1
# 
# fig, ax = plt.subplots(figsize = [10,7])
# ax.set_xticks(list(range(len(x_vars))))
# ax.set_xticklabels(x_vars, rotation = 'vertical')
# 
# ax.plot(df_params.index, df_params['R-squared'], label = 'R-squared', c = 'red')
# ax.legend(loc = 0)
# ax.set_ylabel('R-squared', color = 'red')
# 
# ax2 = ax.twinx()
# ax2.plot(df_params.index, df_params['RSS'], label = 'RSS', c = 'blue')
# ax2.legend(loc = 0)
# ax2.set_ylabel('Residual Sum of Squares', color = 'blue')
# fig.suptitle('OLS Forward Selection')
# plt.show()
# =============================================================================



####################################################################################################
### OLS Linear Regression using Backward Selection w/ K-Fold CV
####################################################################################################
# =============================================================================
# df_data = pd.read_csv(myPath + 'ModelingData.csv', index_col = [0])
# 
# df_mod01 = pd.get_dummies(df_data.drop(['Symbol', 'Date', 'SubIndustry',
#                                         'AnnualRiskPremium', 'AnnualMarketPremium'], 
#                                        axis = 1), drop_first = True)
# col01 = ['RemovedParameter', 'Parameters', 'ParamQty', 'RSS', 'R-squared']
# df_params = pd.DataFrame(columns = col01)
# X_vars = list(df_mod01.columns.drop('AnnualReturn'))
# x_vars = X_vars
# removed = []
# RSS = math.inf
# i = 0
# df_mod01 = df_mod01.sample(frac = 1)
# KFolds = 5
# N = len(X_vars)
# while len(X_vars) != 1:
#     start = len(X_vars)
#     var = ''
#     if i != (N - start):
#         RSS = math.inf
#         Score = 0
#     for x in X_vars:
#         len(x_vars)
#         x_vars = set(X_vars) - set(removed) - set([x])
#         k = 0
#         Score_list = []
#         RSS_list = []
#         while k < KFolds:
#             data = df_mod01.iloc[k * int(len(df_mod01)/KFolds):int(len(df_mod01)/KFolds) * (k + 1)]
#             X_train = df_mod01.drop(data.index)
#             X_train_data = X_train[x_vars]
#             X_test_data = data[x_vars]
#             Y_test = data['AnnualReturn']
#             Y_train = X_train['AnnualReturn']
#             reg = LinearRegression().fit(X_train_data,Y_train)
#             y_pred = reg.predict(X_test_data)
#             RSS_list.append(round(np.sum((Y_test - y_pred)**2), 2))            
#             Score_list.append(round(reg.score(X_test_data, Y_test), 4))
#             k += 1
#         RSS_new = np.mean(RSS_list)
#         Score = np.mean(Score_list)      
#         if RSS_new < RSS:
#             RSS = RSS_new
#             var = x
#             RSS_list_main = RSS_list
#             Score_list_main = Score_list
#     if var != '':
#         removed.append(var)
#         X_vars.remove(var)
#         print('\n--------\n' + str(len(x_vars)))
#         print('RemovedParameters: ' + str(var))
#         print('RSS: ' + str(round(RSS,4)) + ' : ' + str(RSS_list_main))
#         print('R^2: ' + str(round(Score, 4)) + ' : ' + str(Score_list_main))
#         print(str(i) + ' : ' + str(len(X_vars)))
#         row1 = pd.Series([var, x_vars, len(x_vars),round(RSS,4), round(Score, 4)], index = col01)
#         df_params = df_params.append(row1, ignore_index = True)        
#     i += 1
# 
# fig, ax = plt.subplots(figsize = [10,7])
# ax.set_xticks(list(range(N)))
# ax.set_xticklabels(removed, rotation = 'vertical')
#  
# ax.plot(df_params.index, df_params['R-squared'], label = 'R-squared', c = 'red')
# ax.legend(loc = 0)
# ax.set_ylabel('R-squared', color = 'red')
#  
# ax2 = ax.twinx()
# ax2.plot(df_params.index, df_params['RSS'], label = 'RSS', c = 'blue')
# ax2.legend(loc = 0)
# ax2.set_ylabel('Residual Sum of Squares', color = 'blue')
# fig.suptitle('OLS Backward Selection')
# plt.show()
# =============================================================================

####################################################################################################
### Ridge Linear Regression w/ K-Fold CV
####################################################################################################
# =============================================================================
# df_data = pd.read_csv(myPath + 'ModelingData.csv', index_col = [0])
# df_mod01 = pd.get_dummies(df_data.drop(['Symbol', 'Date', 'SubIndustry',
#                                         'AnnualRiskPremium', 'AnnualMarketPremium'], 
#                                        axis = 1), drop_first = True)
# df_mod01 = df_mod01.sample(frac = 1)
# col01 = ['Alpha', 'RSS', 'R-squared']
# df_params = pd.DataFrame(columns = col01)
# #df_mod01 = df_mod01.sample(frac = 1)
# KFolds = 5
# for a in np.arange(0,1,0.05):
#     k = 0
#     Score_list = []
#     RSS_list = []
#     while k < KFolds:
#         df_test = df_mod01.iloc[k * int(len(df_mod01)/KFolds):int(len(df_mod01)/KFolds) * (k + 1)]
#         df_train = df_mod01.drop(df_test.index)        
#         X_test = df_test.drop('AnnualReturn', axis = 1)
#         X_train = df_train.drop('AnnualReturn', axis = 1)
#         Y_test = df_test['AnnualReturn']
#         Y_train = df_train['AnnualReturn']
#         ridge = Ridge(alpha = a).fit(X_train,Y_train)
#         y_pred = ridge.predict(X_test)
#         RSS_list.append(np.sum((Y_test - y_pred)**2))            
#         Score_list.append(ridge.score(X_test, Y_test))
#         k += 1
#     RSS = np.mean(RSS_list)
#     Score = np.mean(Score_list)
#     print('--------')
#     print('Alpha: ' + str(round(a,2)))
#     print('RSS: ' + str(round(RSS,4)))
#     print('R^2: ' + str(round(Score, 4)))
#     row1 = pd.Series([a, round(RSS,4), round(Score, 4)], index = col01)
#     df_params = df_params.append(row1, ignore_index = True)
# 
# df_params.set_index('Alpha', inplace = True)
# fig, ax = plt.subplots(figsize = [10,7])
# ax.plot(df_params.index, df_params['R-squared'], label = 'R-squared', c = 'red')
# ax.legend(loc = 0)
# ax.set_ylabel('R-squared', color = 'red')
# ax.set_xlabel('Alpha')  
# ax2 = ax.twinx()
# ax2.plot(df_params.index, df_params['RSS'], label = 'RSS', c = 'blue')
# ax2.legend(loc = 0)
# ax2.set_ylabel('Residual Sum of Squares', color = 'blue')
# fig.suptitle('Ridge Regression')
# plt.show()
# =============================================================================

####################################################################################################
### Lasso Linear Regression w/ K-Fold CV
####################################################################################################
df_data = pd.read_csv(myPath + 'ModelingData.csv', index_col = [0])
df_mod01 = pd.get_dummies(df_data.drop(['Symbol', 'Date', 'SubIndustry',
                                        'AnnualRiskPremium', 'AnnualMarketPremium'], 
                                       axis = 1), drop_first = True)
col01 = ['Alpha', 'RSS', 'R-squared']
df_params = pd.DataFrame(columns = col01)
#df_mod01 = df_mod01.sample(frac = 1)
KFolds = 5
for a in np.arange(0.001,.125,0.05):
    a = round(a,3)
    k = 0
    Score_list = []
    RSS_list = []
    while k < KFolds:
        df_test = df_mod01.iloc[k * int(len(df_mod01)/KFolds):int(len(df_mod01)/KFolds) * (k + 1)]
        df_train = df_mod01.drop(df_test.index)        
        X_test = df_test.drop('AnnualReturn', axis = 1)
        X_train = df_train.drop('AnnualReturn', axis = 1)
        Y_test = df_test['AnnualReturn']
        Y_train = df_train['AnnualReturn']
        lasso = Lasso(alpha = a, max_iter = 1000).fit(X_train,Y_train)
        y_pred = lasso.predict(X_test)
        RSS_list.append(np.sum((Y_test - y_pred)**2))            
        Score_list.append(lasso.score(X_test, Y_test))
        k += 1
    RSS = np.mean(RSS_list)
    Score = np.mean(Score_list)
    print('--------')
    print('Alpha: ' + str(a))
    print('#Coef: ' + str(np.sum(lasso.coef_ != 0)))
    print('RSS: ' + str(round(RSS,4)))
    print('R^2: ' + str(round(Score, 4)))
    row1 = pd.Series([a, round(RSS,4), round(Score, 4)], index = col01)
    df_params = df_params.append(row1, ignore_index = True)        

df_params.set_index('Alpha', inplace = True)
fig, ax = plt.subplots(figsize = [10,7])
ax.plot(df_params.index, df_params['R-squared'], label = 'R-squared', c = 'red')
ax.legend(loc = 0)
ax.set_ylabel('R-squared', color = 'red')
ax.set_xlabel('Alpha')  
ax2 = ax.twinx()
ax2.plot(df_params.index, df_params['RSS'], label = 'RSS', c = 'blue')
ax2.legend(loc = 0)
ax2.set_ylabel('Residual Sum of Squares', color = 'blue')
fig.suptitle('Lasso Regression')
plt.show()

####################################################################################################
### Logistic Linear Regression using Forward Selection w/ K-Fold CV
####################################################################################################

df_data = pd.read_csv(myPath + 'ModelingData.csv', index_col = [0])
df_mod01 = pd.get_dummies(df_data.drop(['Symbol', 'Date', 'SubIndustry',
                                        'AnnualMarketPremium', 'AnnualRiskPremium'], 
                                       axis = 1))#, drop_first = True)

print(df_data.head())
df_data.columns
dep_var = 'AnnualReturn'
b = []
for i in range(len(df_mod01)):
    if df_mod01[dep_var].iloc[i - 1] <= 1.1:
        b.append(0)
    else:
        b.append(1)
df_mod01[dep_var] = b
len(df_mod01[df_mod01[dep_var] == 1]) / len(df_mod01)
len(df_mod01[df_mod01[dep_var] == 0]) / len(df_mod01)
len(df_mod01)
col01 = ['Parameters', 'ParamQty', 'Accuracy', 'Precision', 'Recall']
df_params = pd.DataFrame(columns = col01)
X_vars = list(df_mod01.columns.drop(dep_var))
x_vars = []
KFolds = 5
i = 0
Score = 0
df_mod01 = df_mod01.sample(frac = 1)
N = len(X_vars)
while len(X_vars) != 0:
    RSS = math.inf
    Score = 0
    recall = 0
    var = ''
    for x in X_vars:
        k = 0
        Score_list = []
        confmat_sum = np.zeros((2,2))
        while k < KFolds:
            X_test = df_mod01.iloc[k * int(len(df_mod01)/KFolds):int(len(df_mod01)/KFolds) * (k + 1)]
            X_train = df_mod01.drop(X_test.index)
            X_train_data = X_train[[x] + x_vars]
            X_test_data = X_test[[x] + x_vars]
            Y_test = X_test[dep_var]
            Y_train = X_train[dep_var]
            logit = LogisticRegression(solver='lbfgs').fit(X_train_data,Y_train)
            y_pred = logit.predict(X_test_data)
            Score_list.append(logit.score(X_test_data, Y_test))
            confmat_sum = confmat_sum + confusion_matrix(Y_test, y_pred)
            k += 1
        Score_new = np.mean(Score_list)
        confmat_new = confmat_sum / KFolds
        TN = confmat_new[0][0]
        TP = confmat_new[1][1]
        FP = confmat_new[0][1]
        FN = confmat_new[1][1]
        accuracy_new = (TN+TP)/(TN+TP+FP+FN)
        precision_new = TP / (TP + FP)
        recall_new = TP + (TP + FN)
        if recall_new > recall:
            accuracy = accuracy_new
            precision = precision_new
            recall = recall_new
            var = x
            Score_list_main = Score_list
            confmat_main = confmat_new
    x_vars.append(var)
    X_vars.remove(var)
    print('\n--------\n' + str(len(x_vars)))
    print('Parameters: ' + str(x_vars))
    print('Accuracy: ' + str(round(Score, 4)) + ' : ' + str(Score_list_main))
    print(str(i) + ' : ' + str(len(X_vars)))
    print('Confusion Matrix: \t' + str(confmat_main))
    row1 = pd.Series([var, len(x_vars), accuracy, precision, recall], index = col01)
    df_params = df_params.append(row1, ignore_index = True)
    i += 1

