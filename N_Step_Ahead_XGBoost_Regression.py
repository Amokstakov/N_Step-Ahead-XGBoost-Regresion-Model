# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 12:49:41 2019

@author: Alexander.Moksyakov
"""

###Imports
import numpy as np
import pandas as pd
import xgboost as xgboost
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from sklearn.metrics import max_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score
from sklearn.preprocessing import normalize
from sklearn.preprocessing import MinMaxScaler

###Data & Data Prerp
report_db = pd.read_csv(r'2020 Report_DBCopy.csv', parse_dates=[0])
cp_db = pd.read_csv(r'CP_Categories.csv')

cjmp_db = cp_db.drop(columns = ['Reference period'])

report_db = report_db.set_index('DATE')


report_db.replace([np.inf, -np.inf], np.nan, inplace=True)
report_db.dropna(axis=1,inplace=True)

cols = report_db.columns
index = report_db.index

x = report_db.values #returns a numpy array
min_max_scaler = MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
report_db = pd.DataFrame(x_scaled)

report_db.columns= cols
report_db.index = index

pca = PCA(n_components = 10, svd_solver='full')
PCA_X = pca.fit_transform(report_db)
#print(np.sum(pca.explained_variance_))

data = pd.DataFrame(PCA_X).set_index(report_db.index)
labels = cp_db[['Food purchased from restaurants 4']].set_index(report_db.index)

def split_data(data, split_date):
    return data[data.index <= split_date].copy(), \
           data[data.index >  split_date].copy()

#x = data.values #returns a numpy array
#min_max_scaler = MinMaxScaler()
#x_scaled = min_max_scaler.fit_transform(x)
#data = pd.DataFrame(x_scaled).set_index(report_db.index)

def build_data(X, y, n_steps_back, n_steps_ahead=1):
    n_steps_ahead = n_steps_ahead-1
    newX = X[:(len(X)-n_steps_back-n_steps_ahead)]
    newX = np.column_stack([newX, y[:(len(X)-n_steps_back-n_steps_ahead)]])
    for i in range(1, n_steps_back):
        newX = np.column_stack([newX, X[i:(len(X)+i-n_steps_back-n_steps_ahead)]])
        newX = np.column_stack([newX, y[i:(len(X)+i-n_steps_back-n_steps_ahead)]])
    newy = y[(n_steps_back+n_steps_ahead):]
    if isinstance(X, pd.DataFrame):
        shifted_index = X.index[(n_steps_back+n_steps_ahead):]
        newX = pd.DataFrame(newX, index=shifted_index)
        newy = pd.DataFrame(newy, index=shifted_index)
    return(newX, newy)

# Predictive model that will predict n_steps_ahead in the future
def n_steps_ahead_predictor(data, labels, n_steps_ahead=1):
    # Construct the training and testing data
    shifted_index = list(data.index[n_steps_ahead:])
    data = pd.DataFrame(data.iloc[:-n_steps_ahead].values, index=shifted_index)
    labels = labels.iloc[n_steps_ahead:].values - labels.iloc[:-n_steps_ahead].values
    labels = pd.DataFrame(labels, index=shifted_index)
    data, labels = build_data(data, labels, n_steps_back=24)
    # Create the splits
    X_train, X_test = split_data(data, '01-Sep-2010')
    y_train, y_test = split_data(labels, '01-Sep-2010')
    #train, test = split_data(full_db, '01-Jan-2018')
    #y_prev_test = y_prev_vals[-len(X_test):].values.reshape(-1,)
    #y_prev_train = y_prev_vals[:len(X_train)].values.reshape(-1,)
    # Create training data
    dtrain = xgboost.DMatrix(X_train.to_numpy(), label=y_train)
    dtest = xgboost.DMatrix(X_test.to_numpy(), label=y_test)
    # Create the model
    num_round = 100000
    evallist = [(dtrain, 'train'), (dtest, 'eval')]
    param = {
                'max_depth': 8,
                'eta': 0.003,
                'verbosity': 1,
                'objective': 'reg:linear', #'reg:squarederror',
                'eval_metric': 'mae', #'rmse',
                'subsample':0.5,
                'gamma': 2,
                'nthread': -1
            }
    reg = xgboost.train(
            param, dtrain, num_round, evallist, 
            early_stopping_rounds=50, verbose_eval=50
        )
    X_test_pred = reg.predict(dtest, ntree_limit=reg.best_ntree_limit)
    return(reg, X_test_pred[n_steps_ahead-1])

predictions = []
for i in range(1,105):
    mod, y_pred = n_steps_ahead_predictor(data, labels, n_steps_ahead=i)
    predictions.append(y_pred)


def plot_performance(X_test, X_test_pred, date_from, date_to, y_test=None, title=None):
    plt.figure(figsize=(15,3))
    if title == None:
        plt.title('From {0} To {1}'.format(date_from, date_to))
    else:
        plt.title(title)
    plt.xlabel('time')
    if y_test is not None:
        plt.plot(y_test.index, y_test, label='actuals')
    plt.plot(X_test.index, X_test_pred, label='predictions')
    plt.legend()
    plt.xlim(left=date_from, right=date_to)


plot_performance(
    data.iloc[-len(predictions):], list(labels.values.reshape(-1,))[-len(predictions)-1] + predictions,
    data.index[0].date(), data.index[-1].date(),
    y_test = labels, 
    title = 'Original and Predicted Data'
)

plot_performance(
    data.iloc[-len(predictions):], list(labels.values.reshape(-1,))[-len(predictions)-1] + predictions,
    data.iloc[-len(predictions):].index[0].date(), data.iloc[-len(predictions):].index[-1].date(),
    y_test = labels.iloc[-len(predictions):], 
    title = 'Original and Predicted Data'
)

plt.plot(data.iloc[-len(predictions):], list(labels.values.reshape(-1,))[-len(predictions)-1] + predictions)
plt.show()


some_test_ = list(labels.values.reshape(-1,))[-len(predictions)-1] + predictions
some_test_ = pd.DataFrame(some_test_)

some_labels = labels[-104:]

some_test_.index = some_labels.index


acc = mean_absolute_error(some_test_, some_labels)
print(acc)

acc = explained_variance_score(some_test_, some_labels)
print(acc)


fig,ax = plt.subplots()
ax = some_labels.plot(ax=ax)
some_test_.plot(ax=ax)
plt.show()


from scipy.stats import sem, t
from scipy import mean
confidence = 0.95
data_test = some_test_

n = len(data_test)
m = mean(data_test)
std_err = sem(data_test)
h = std_err * t.ppf((1 + confidence) / 2, n - 1)

print(h)

some_test_upper = some_test_ + h
some_test_lower = some_test_ - h

some_test_upper = some_test_upper[0]
some_test_lower = some_test_lower[0]

fig,ax = plt.subplots()
ax = some_labels.plot(ax=ax)
some_test_.plot(ax=ax)
some_test_upper.plot(ax=ax, color='yellow')
some_test_lower.plot(ax=ax, color='yellow')
plt.fill_between(some_labels.index,some_test_upper.values, some_test_lower.values, facecolor='yellow', alpha=0.5)
plt.show()


def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mean_absolute_percentage_error(some_labels,some_test_)

#120
'''
n_steps_ahead = 12
data = data.iloc[n_steps_ahead:]
y_prev_vals = pd.DataFrame(labels.iloc[:-n_steps_ahead].values, index=data.index)
labels = labels.iloc[n_steps_ahead:].values - labels.iloc[:-n_steps_ahead].values
labels = pd.DataFrame(labels, index=data.index)
data, labels = build_data(data, labels, n_steps_back=24)
y_prev_vals = y_prev_vals.iloc[-len(labels):]

X_train, X_test = split_data(data, '01-Jan-2016')
y_train, y_test = split_data(labels, '01-Jan-2016')
#train, test = split_data(full_db, '01-Jan-2018')

y_prev_test = y_prev_vals[-len(X_test):].values.reshape(-1,)
y_prev_train = y_prev_vals[:len(X_train)].values.reshape(-1,)

#plt.figure(figsize=(15,5))
#plt.xlabel('time')
#plt.plot(X_train.index,y_train['Food 4'])
#plt.plot(X_test.index,y_test['Food 4'])
#plt.show()

dtrain = xgboost.DMatrix(X_train.to_numpy(), label=y_train)
dtest = xgboost.DMatrix(X_test.to_numpy(), label=y_test)

num_round = 100000
evallist = [(dtrain, 'train'), (dtest, 'eval')]
param = {
            'max_depth': 8,
            'eta': 0.003,
            'verbosity': 1,
            'objective': 'reg:linear', #'reg:squarederror',
            'eval_metric': 'mae', #'rmse',
            'subsample':0.5,
            'gamma': 2,
            'nthreads': -1
        }

reg = xgboost.train(param, dtrain, num_round, evallist, early_stopping_rounds=50, verbose_eval=True)

xgboost.plot_importance(reg, max_num_features=10)

def plot_performance(X_test, X_test_pred, date_from, date_to, y_test=None, title=None):
    plt.figure(figsize=(15,3))
    if title == None:
        plt.title('From {0} To {1}'.format(date_from, date_to))
    else:
        plt.title(title)
    plt.xlabel('time')
    if y_test is not None:
        plt.plot(y_test.index, y_test, label='actuals')
    plt.plot(X_test.index, X_test_pred, label='predictions')
    plt.legend()
    plt.xlim(left=date_from, right=date_to)
    

X_test_pred = reg.predict(dtest, ntree_limit=reg.best_ntree_limit)

plot_performance(
    X_test, y_prev_test + X_test_pred, 
    #X_test, y_prev_test[0] + np.cumsum(X_test_pred),
    data.index[0].date(), data.index[-1].date(),
    y_test = y_prev_vals+labels, 
    title = 'Original and Predicted Data'
)

plot_performance(
    X_test, y_prev_test + X_test_pred, 
    y_test.index[0].date(), y_test.index[-1].date(),
    y_test = pd.DataFrame(y_prev_test + y_test.values.reshape(-1,), index=X_test.index),
    title = 'Forecasted Original and Predicted Data'
)

plt.legend()
plt.show()
'''