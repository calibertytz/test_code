import open_competition as op
from open_competition.tabular.model_fitter import LGBFitter
import pandas as pd
from dataclasses import dataclass, asdict
from sklearn.model_selection import KFold
from dataclasses import dataclass, asdict
import hyperopt.pyll
from hyperopt import fmin, tpe, hp
import copy
import numpy as np
from tqdm import tqdm
import sys
import os

output_path = 'param_range'
if not os.path.exists(output_path):
    os.mkdir(output_path)

# load data
train_path = '../toy_data/train_onehot.csv'
test_path = '../toy_data/test_onehot.csv'
df_test = pd.read_csv(test_path)
df_train = pd.read_csv(train_path)

cpu_count = 8

'''
Firstly, we test num_round,
for gbdt, suitable num_round is 2000.
for dart, suitable num_round need large than 2000 (need large learning_rate)
for goss, suitable num_round is 2000

we fix num_round as 2000

then for learning rate, we fix learning_rate as 5e-2.

for dart, learning_rate should be 7e-2

then for num_leaves, 

for gbdt, max is 96 fix as 32

for dart, 

for goss, if num_leaves is large, the needed num_round is small. max is 64 fix as 32

for feature_fraction

for gbdt, 0.2-1.0 fix as 0.6

for goss 0.2 - 0.8 fix as 0.7


for bagging_fraction

for max_depth,
gbdt is 5

gbdt 2000, 5e-2, 32, 5, 0.6, 
dart 2000 7e-2, 32, 
goss 2000 5e-2, 32, 5, 0.7

'''

'''
for has extra tree
gbdt: lr:0.07, max_depth:7, num_leaves: 32, 
goss: lr:0.05, max_depth: , num_leaves: 
dart:
rf: 



'''

num_leaves = 32
num_round = 2000
learning_rate = 1e-2
feature_fraction = 0.8
bagging_fraction = 0.8
bagging_freq = None
for num_leaves in [16, 32, 64, 96, 128, 164, 192]:
    common_params = {'num_thread': 32,
                     'num_leaves': num_leaves,
                     'metric': 'binary_error',
                     'objective': 'binary',
                     'num_round': num_round,
                     'learning_rate': learning_rate,
                     'feature_fraction': feature_fraction,
                     'bagging_fraction': bagging_fraction,
                     'bagging_freq': bagging_freq}

    model_fitter = LGBFitter(label='label')
    kfold = KFold(n_splits=5)

    _,_,acc,_ = model_fitter.train_k_fold(kfold, df_train, df_test, params=common_params)
    print(common_params)
    print(np.mean(acc))