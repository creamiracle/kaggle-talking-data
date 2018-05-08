# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split  
import gc
import xgboost as xgb
import operator  

dtypes = {
        'ip'                          : 'uint32',
        'app'                         : 'uint16',
        'device'                      : 'uint16',
        'os'                          : 'uint16',
        'channel'                     : 'uint16',
        'is_attributed'               : 'float32',
        'click_id'                    : 'uint32',
        'day'                         : 'uint8',
        'hour'                        : 'uint8',
        'ip_count'                    : 'uint16',
        'app_count'                   : 'uint16',
        'user_count'                  : 'uint16',
        'os_count'                    : 'uint16',
        'channel_count'               : 'uint16',
        'n_channels'                  : 'uint16',
        'ip_day_ch_hour_var'          : 'float32',
        'ip_app_day_hour_count'       : 'uint16',
        'ip_app_count'                : 'uint16',
        'ip_app_os_count'             : 'uint16',
        'ip_day_channel_var'          : 'float32',
        'ip_app_os_var'               : 'float32',
        'ip_app_channel_var_day'      : 'float32',
        'ip_app_channel_count_day'    : 'uint16',
        'ip_app_channel_mean_hour'    : 'float32',
        'ip_dev'                      : 'uint32',
        'ip_tcount'                   : 'uint16',
        'nip_day_test_hh'             : 'uint32',
        'X0'                          : 'uint16',
        'X1'                          : 'uint32',
        'X2'                          : 'uint16',
        'X3'                          : 'uint16',
        'X4'                          : 'uint16',
        'X5'                          : 'uint16',
        'X6'                          : 'uint16',
        'X7'                          : 'uint32',
        'X8'                          : 'uint16',
        'UsrNewness'                  : 'uint32',
        'nextClick'                   : 'float32',
        'nextClick_shift'             : 'float32',
        'time_delta_user'             : 'float32',
        'time_delta_ip'               : 'float32',
        'time_delta_ip_app'           : 'float32',
        'time_delta_ip_cha'           : 'float32',
        'time_delta_ip_dev'           : 'float32'
        }


print("Loading Data")
skip = range(1,131886952)
train = pd.read_csv('train_after.csv',dtype=dtypes, skiprows = skip, header=0)
test = pd.read_csv('test.csv', header=0)

len_train = len(train) - len(test)
#len_train = 184903890
print("len_train: ",len_train)


#random split for both 
print("Split validation!")
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

train.info()
del train
del train_
gc.collect()

print("Preparing the datasets for training...")
params = {
          'tree_method' : "approx", 
          'objective' : 'binary:logistic', 
          'eval_metric' : 'auc', 
          'random_state' : 99,
          'silent' : True,
          'colsample_bylevel': 0.1,
          'colsample_bytree': 1.0,
          'gamma': 5.103973694670875e-08,
          'learning_rate': 0.140626707498132,
          'max_delta_step': 20,
          'max_depth': 6,
          'min_child_weight': 4,
          'n_estimators': 100,
          'reg_alpha': 1e-09,
          'reg_lambda': 1000.0,
          'scale_pos_weight': 499.99999999999994,
          'subsample': 0.1,
          }

dtrain = xgb.DMatrix(train_X, train_y)
del train_X, train_y
gc.collect()

dvalid = xgb.DMatrix(val_X, val_y)
del val_X, val_y
gc.collect()

watchlist = [(dtrain, 'train'), (dvalid, 'valid')]

print("Training model!")
model = xgb.train(params, dtrain, 1000, watchlist, maximize=True, early_stopping_rounds=20, verbose_eval=10)
del dtrain
del dvalid
gc.collect()

def ceate_feature_map(features):  
    outfile = open('xgb.fmap', 'w')  
    i = 0  
    for feat in features:  
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))  
        i = i + 1  
    outfile.close() 

print("saveing feat_importance!")
features = [x for x in train_.columns if x not in ['id','loss']]  
ceate_feature_map(features)  
  
importance = model.get_fscore(fmap='xgb.fmap')  
importance = sorted(importance.items(), key=operator.itemgetter(1))  
  
df = pd.DataFrame(importance, columns=['feature', 'fscore'])  
df['fscore'] = df['fscore'] / df['fscore'].sum()  

plt.figure()  
df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 10))  
plt.title('XGBoost Feature Importance')  
plt.xlabel('relative importance')  
plt.savefig("feat_importance.jpg",dpi = 300)

print("predicting...")
sub = pd.read_csv('test.csv', dtype='int', usecols=['click_id'])
sub['is_attributed'] = model.predict(xgb.DMatrix(test), ntree_limit=model.best_ntree_limit)
sub.to_csv('sub_xgb.csv',index=False)

print("All done...")