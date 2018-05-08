# coding: utf-8

import pandas as pd
#import lightgbm as lgb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split  
import gc
import xgboost as xgb
from skopt import BayesSearchCV


train_cols = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed']
test_cols = ['ip', 'app', 'device', 'os', 'channel', 'click_time']

dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8',
        'click_id'      : 'uint32'
        }

#skip = range(1,2)
print("Loading Data")
train = pd.read_csv('train.csv',  dtype=dtypes,
        header=0,usecols=train_cols,parse_dates=["click_time"])#.sample(1000)
test = pd.read_csv('test.csv', dtype=dtypes, header=0,
        usecols=test_cols,parse_dates=["click_time"])#.sample(1000)

#68941878(2017-11-08 00:00:00) [68941878:131886952]
#131886953(2017-11-09 00:00:00)[131886953:]
#62945076   2017-11-08 23:59:59
#62945077   2017-11-09 00:00:00

len_train = len(train)
print('The initial size of the train set is', len_train)
print('Binding the training and test set together...')
train=train.append(test)
del test
gc.collect()

print("Creating new time features: 'hour' and 'day'...")
train['hour'] = train["click_time"].dt.hour.astype('uint8')
train['day'] = train["click_time"].dt.day.astype('uint8')

print("Creating new time features: 'ip_count'...")
n_chans = train[['ip','channel']].groupby(by=['ip'])[['channel']].count().reset_index().rename(columns={'channel': 'ip_count'})
print('Merging the channels data with the main data set...')
train = train.merge(n_chans, on=['ip'], how='left')
del n_chans
gc.collect()

print("Creating new time features: 'app_count'...")
n_chans = train[['app','channel']].groupby(by=['app'])[['channel']].count().reset_index().rename(columns={'channel': 'app_count'})
print('Merging the channels data with the main data set...')
train = train.merge(n_chans, on=['app'], how='left')
del n_chans
gc.collect()

print("Creating new time features: 'user_count'...")
n_chans = train[['app','ip','device','os','channel']].groupby(by=['app','ip','device','os'])[['channel']].count().reset_index().rename(columns={'channel': 'user_count'})
print('Merging the channels data with the main data set...')
train = train.merge(n_chans, on=['app','ip','device','os'], how='left')
del n_chans
gc.collect()

print("Creating new time features: 'os_count'...")
n_chans = train[['ip','device','os','channel']].groupby(by=['ip','device','os'])[['channel']].count().reset_index().rename(columns={'channel': 'os_count'})
print('Merging the channels data with the main data set...')
train = train.merge(n_chans, on=['ip','device','os'], how='left')
del n_chans
gc.collect()

print("Creating new time features: 'channel_count'...")
n_chans = train[['app','channel']].groupby(by=['channel'])[['app']].count().reset_index().rename(columns={'app': 'channel_count'})
print('Merging the channels data with the main data set...')
train = train.merge(n_chans, on=['channel'], how='left')
del n_chans
gc.collect()

print("Creating new count features: 'n_channels'")
n_chans = train[['ip','day','hour','channel']].groupby(by=['ip','day',
          'hour'])[['channel']].count().reset_index().rename(columns={'channel': 'n_channels'})
print('Merging the channels data with the main data set...')
train = train.merge(n_chans, on=['ip','day','hour'], how='left')
del n_chans
gc.collect()

print("Creating new count features: 'ip_day_ch_hour_var'")
n_chans = train[['ip','day','hour','channel']].groupby(by=['ip','day',
          'channel'])[['hour']].var().reset_index().rename(columns={'hour': 'ip_day_ch_hour_var'})
print('Merging the channels data with the main data set...')
train = train.merge(n_chans, on=['ip','day','channel'], how='left')
del n_chans
gc.collect()

print("Creating new count features: 'ip_app_day_hour_count'")
n_chans = train[['ip','app','day','hour','channel']].groupby(by=['ip','day','hour', 
          'app'])[['channel']].count().reset_index().rename(columns={'channel': 'ip_app_day_hour_count'})
print('Merging the channels data with the main data set...')
train = train.merge(n_chans, on=['ip','day','hour','app'], how='left')
del n_chans
gc.collect()

print("Creating new count features: 'ip_app_count'")
n_chans = train[['ip','app', 'channel']].groupby(by=['ip', 
          'app'])[['channel']].count().reset_index().rename(columns={'channel': 'ip_app_count'})
print('Merging the channels data with the main data set...')
train = train.merge(n_chans, on=['ip','app'], how='left')
del n_chans
gc.collect()

print("Creating new count features: 'ip_app_os_count'")
n_chans = train[['ip','app', 'os', 'channel']].groupby(by=['ip', 'app', 
          'os'])[['channel']].count().reset_index().rename(columns={'channel': 'ip_app_os_count'})
print('Merging the channels data with the main data set...')       
train = train.merge(n_chans, on=['ip','app', 'os'], how='left')
del n_chans
gc.collect()

print("Creating new count features: 'ip_day_channel_var'")
gp = train[['ip','day','hour','channel']].groupby(by=['ip',
        'day','channel'])[['hour']].var().reset_index().rename(index=str, columns={'hour': 'ip_day_channel_var'})
print('Merging the channels data with the main data set...')   
train = train.merge(gp, on=['ip','day','channel'], how='left')
del gp
gc.collect()

print("Creating new count features: 'ip_app_os_var'")
gp = train[['ip','app', 'os', 'hour']].groupby(by=['ip', 
    'app', 'os'])[['hour']].var().reset_index().rename(index=str, columns={'hour': 'ip_app_os_var'})
print('Merging the channels data with the main data set...')   
train = train.merge(gp, on=['ip','app', 'os'], how='left')
del gp
gc.collect()

print("Creating new count features: 'ip_app_channel_var_day'")
gp = train[['ip','app', 'channel', 'day']].groupby(by=['ip', 
    'app', 'channel'])[['day']].var().reset_index().rename(index=str, columns={'day': 'ip_app_channel_var_day'})
print('Merging the channels data with the main data set...')   
train = train.merge(gp, on=['ip','app','channel'], how='left')
del gp
gc.collect()

print("Creating new count features: 'ip_app_channel_count_day'")
gp = train[['ip','app', 'channel', 'day']].groupby(by=['ip', 
    'app', 'channel'])[['day']].count().reset_index().rename(index=str, columns={'day': 'ip_app_channel_count_day'})
print('Merging the channels data with the main data set...')   
train = train.merge(gp, on=['ip','app','channel'], how='left')
del gp
gc.collect()

print("Creating new count features: 'ip_app_channel_mean_hour'")
gp = train[['ip','app', 'channel','hour']].groupby(by=['ip', 
    'app', 'channel'])[['hour']].mean().reset_index().rename(index=str, columns={'hour': 'ip_app_channel_mean_hour'})
print('Merging the channels data with the main data set...')   
train = train.merge(gp, on=['ip','app', 'channel'], how='left')
del gp
gc.collect()

print("Creating new count features: 'ip_dev'")
gp = train[['ip', 'device', 'hour', 'channel']].groupby(by=['ip', 'device', 
            'hour'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_dev'})
print('Merging the channels data with the main data set...')   
train = train.merge(gp, on=['ip','device','hour'], how='left')
del gp
gc.collect()

print('grouping by ip-day-hour combination...')
gp = train[['ip','day','hour','channel']].groupby(by=['ip','day',
                'hour'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_tcount'})
train = train.merge(gp, on=['ip','day','hour'], how='left')
del gp
gc.collect()

print("Creating new count features: 'in_test_hh'")
most_freq_hours_in_test_data = [4, 5, 9, 10, 13, 14]
least_freq_hours_in_test_data = [6, 11, 15]
train['in_test_hh'] = (3 - 2*train['hour'].isin(most_freq_hours_in_test_data) - 1*train['hour'].isin( least_freq_hours_in_test_data ) ).astype('uint8')
gp = train[['ip', 'day', 'in_test_hh', 'channel']].groupby(by=['ip', 'day', 'in_test_hh'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'nip_day_test_hh'})
print('Merging the channels data with the main data set...')   
train = train.merge(gp, on=['ip','day','in_test_hh'], how='left')
train.drop(['in_test_hh'], axis=1, inplace=True)
del gp
gc.collect()

print("Creating new count features: 'X'")
naddfeat=9
for i in range(0,naddfeat):
    if i==0: selcols=['ip', 'channel']; QQ=4;
    if i==1: selcols=['ip', 'device', 'os', 'app']; QQ=5;
    if i==2: selcols=['ip', 'day', 'hour']; QQ=4;
    if i==3: selcols=['ip', 'app']; QQ=4;
    if i==4: selcols=['ip', 'app', 'os']; QQ=4;
    if i==5: selcols=['ip', 'device']; QQ=4;
    if i==6: selcols=['app', 'channel']; QQ=4;
    if i==7: selcols=['ip', 'os']; QQ=5;
    if i==8: selcols=['ip', 'device', 'os', 'app']; QQ=4;
    print('selcols',selcols,'QQ',QQ)
        
    if QQ==4:
        gp = train[selcols].groupby(by=selcols[0:len(selcols)-1])[selcols[len(selcols)-1]].nunique().reset_index().rename(index=str, columns={selcols[len(selcols)-1]: 'X'+str(i)})
        train = train.merge(gp, on=selcols[0:len(selcols)-1], how='left')
    if QQ==5:
        gp = train[selcols].groupby(by=selcols[0:len(selcols)-1])[selcols[len(selcols)-1]].cumcount() + 1
        train['X'+str(i)]=gp.values
    
    del gp
    gc.collect()

# print("Creating new count features: 'UsrappNewness'")
# train['UsrappNewness'] = train.groupby(['ip','app', 'device', 'os']).cumcount() + 1
# gc.collect()

print("Creating new count features: 'UsrNewness'")
train['UsrNewness'] = train.groupby(['ip', 'device', 'os']).cumcount() + 1
gc.collect()

print("Creating new count features: 'nextClick'")
new_feature = 'nextClick'
D=2**26
train['category'] = (train['ip'].astype(str) + "_" + train['app'].astype(str) + "_" + train['device'].astype(str) \
        + "_" + train['os'].astype(str)).apply(hash) % D
click_buffer= np.full(D, 3000000000, dtype=np.uint32)
train['epochtime']= train['click_time'].astype(np.int64) // 10 ** 9
next_clicks= []
for category, t in zip(reversed(train['category'].values), reversed(train['epochtime'].values)):
    next_clicks.append(click_buffer[category]-t)
    click_buffer[category]= t
del(click_buffer)
QQ= list(reversed(next_clicks))
train.drop(['epochtime','category'], axis=1, inplace=True)
train[new_feature] = pd.Series(QQ).astype('float32')
#predictors.append(new_feature)
train[new_feature+'_shift'] = train[new_feature].shift(+1).values
#predictors.append(new_feature+'_shift')
del QQ, next_clicks
gc.collect()

print("Creating new count features: 'time_delta_user'")
train['time_delta_user'] = train.sort_values(['click_time']).groupby(by=['ip','device', 'os'])[['click_time']].diff().astype('timedelta64[s]')

print("Creating new count features: 'time_delta_user_app'")
train['time_delta_ip'] = train.sort_values(['click_time']).groupby(by=['ip'])[['click_time']].diff().astype('timedelta64[s]')

print("Creating new count features: 'time_delta_user_app'")
train['time_delta_ip_app'] = train.sort_values(['click_time']).groupby(by=['ip','app'])[['click_time']].diff().astype('timedelta64[s]')

print("Creating new count features: 'time_delta_user_app'")
train['time_delta_ip_cha'] = train.sort_values(['click_time']).groupby(by=['ip','channel'])[['click_time']].diff().astype('timedelta64[s]')

print("Creating new count features: 'time_delta_user_app'")
train['time_delta_ip_dev'] = train.sort_values(['click_time']).groupby(by=['ip','device'])[['click_time']].diff().astype('timedelta64[s]')

#Change Here When Change feature
print("Adjusting the data types of the new count features... ")
train['n_channels'] = train['n_channels'].astype('uint16')
train['ip_app_count'] = train['ip_app_count'].astype('uint16')
train['ip_app_os_count'] = train['ip_app_os_count'].astype('uint16')
train['ip_dev'] = train['ip_dev'].astype('uint32')
train['nip_day_test_hh'] = train['nip_day_test_hh'].astype('uint32')
train['ip_app_day_hour_count'] = train['ip_app_day_hour_count'].astype('uint16')
train['app_count'] = train['app_count'].astype('uint16')
train['channel_count'] = train['channel_count'].astype('uint16')
train['ip_app_channel_count_day'] = train['ip_app_channel_count_day'].astype('uint16')
train['ip_count'] = train['ip_count'].astype('uint16')
train['user_count'] = train['user_count'].astype('uint16')
train['os_count'] = train['os_count'].astype('uint16')
train['UsrNewness'] = train['UsrNewness'].astype('uint16')
#train['UsrappNewness'] = train['UsrappNewness'].astype('uint16')
train['ip_tcount'] = train['ip_tcount'].astype('uint16')

train = train.drop(['click_time'], axis = 1)
train.info()


#random split for both 
test = train[len_train:]
train_ = train[:len_train]
target = 'is_attributed'
target = train_[target]
train_ = train_.drop(['is_attributed'],axis = 1)
train_X,val_X, train_y, val_y = train_test_split(train_,target,test_size = 0.1,random_state = 0) 
train_y = train_y.astype('uint8')
val_y = val_y.astype('uint8')
print('The size of the test set is ', len(test))
print('The size of the validation set is ', len(val_X))
print('The size of the train set is ', len(train_X))

del train
gc.collect()

print("Tuning!")
# tuning
ITERATIONS = 1000 # 1000
# Classifier
bayes_cv_tuner = BayesSearchCV(
    estimator = xgb.XGBClassifier(
        n_jobs = 1,
        objective = 'binary:logistic',
        eval_metric = 'auc',
        silent=1,
        tree_method='approx'
    ),
    search_spaces = {
        'learning_rate': (0.01, 1.0, 'log-uniform'),
        'min_child_weight': (0, 10),
        'max_depth': (0, 50),
        'max_delta_step': (0, 20),
        'subsample': (0.01, 1.0, 'uniform'),
        'colsample_bytree': (0.01, 1.0, 'uniform'),
        'colsample_bylevel': (0.01, 1.0, 'uniform'),
        'reg_lambda': (1e-9, 1000, 'log-uniform'),
        'reg_alpha': (1e-9, 1.0, 'log-uniform'),
        'gamma': (1e-9, 0.5, 'log-uniform'),
        'min_child_weight': (0, 5),
        'n_estimators': (50, 100),
        'scale_pos_weight': (1e-6, 500, 'log-uniform')
    },    
    scoring = 'roc_auc',
    cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=42
    ),
    n_jobs = 3,
    n_iter = ITERATIONS,   
    verbose = 0,
    refit = True,
    random_state = 42
)

result = bayes_cv_tuner.fit(train_.values, target.values, callback=None)

del train_
gc.collect()

print("Preparing the datasets for training...")
params = {
          'tree_method' : "approx", 
          'objective' : 'binary:logistic', 
          'eval_metric' : 'auc', 
          'random_state' : 99,
          'silent' : True
          }

print("Updating parameters!")
params.update(result.best_params_)

dtrain = xgb.DMatrix(train_X, train_y)
del train_X, train_y
gc.collect()

dvalid = xgb.DMatrix(val_X, val_y)
del val_X, val_y
gc.collect()

watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
model = xgb.train(params, dtrain, 2000, watchlist, maximize=True, early_stopping_rounds=50, verbose_eval=10)
del dtrain
del dvalid
gc.collect()

print("predicting...")
sub = pd.read_csv('test.csv', dtype='int', usecols=['click_id'])
sub['is_attributed'] = model.predict(xgb.DMatrix(test), ntree_limit=model.best_ntree_limit)
sub.to_csv('sub_xgb_delta.csv',index=False)

print("All done...")