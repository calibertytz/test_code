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

num_leaves = 32
num_round = 2000
learning_rate = 5e-2

boosting_mode = sys.argv[1]  # 'gbdt', 'dart', 'goss'
extra_tree = sys.argv[2]  # 0 denote False, 1 denote True

common_params = {'num_thread': 32,
                 'num_leaves': num_leaves,
                 'metric': 'binary_error',
                 'objective': 'binary',
                 'num_round': num_round,
                 'learning_rate': learning_rate,
                 'feature_fraction': 0.6,
                 'bagging_fraction': 0.8}

common_params['boosting'] = boosting_mode

if extra_tree:
    common_params['extra_trees'] = True
else:
    pass

res_list = {}
for lr in [3e-2, 5e-2, 7e-2, 9e-2]:
    params = common_params.copy()
    params['learning_rate'] = lr
    kfold = KFold(n_splits=5)
    fitter = LGBFitter()
    _, _, res, _ = fitter.train_k_fold(kfold, df_train, df_test, params)
    res_list[lr] = res
    print(f'lr: {lr}, res: {res}')
pd.DataFrame(res_list).to_csv(f'param_range/res_{boosting_mode}_{extra_tree}_lr.csv', index=False)

res_list = {}
for max_depth in [3, 5, 7]:
    params = common_params.copy()
    params['max_depth'] = max_depth
    kfold = KFold(n_splits=5)
    fitter = LGBFitter()
    _, _, res, _ = fitter.train_k_fold(kfold, df_train, df_test, params)
    res_list[max_depth] = res
    print(f'lr: {max_depth}, res: {res}')
pd.DataFrame(res_list).to_csv(f'param_range/res_{boosting_mode}_{extra_tree}_max_depth.csv', index=False)


res_list = {}
for num_leaves in [16, 32, 64, 96]:
    params = common_params.copy()
    params['num_leaves'] = num_leaves
    kfold = KFold(n_splits=5)
    fitter = LGBFitter()
    _, _, res, _ = fitter.train_k_fold(kfold, df_train, df_test, params)
    res_list[num_leaves] = res
    print(f'lr: {num_leaves}, res: {res}')
pd.DataFrame(res_list).to_csv(f'param_range/res_{boosting_mode}_{extra_tree}_num_leaves.csv', index=False)

res_list = {}
for p in [0.2, 0.4, 0.6, 0.8]:
    params = common_params.copy()
    params['feature_fraction'] = p
    kfold = KFold(n_splits=5)
    fitter = LGBFitter()
    _, _, res, _ = fitter.train_k_fold(kfold, df_train, df_test, params)
    res_list[p] = res
    print(f'lr: {p}, res: {res}')
pd.DataFrame(res_list).to_csv(f'param_range/res_{boosting_mode}_{extra_tree}_feature_fraction.csv', index=False)

res_list = {}
for p in [0.2, 0.4, 0.6, 0.8]:
    params = common_params.copy()
    params['bagging_fraction'] = p
    kfold = KFold(n_splits=5)
    fitter = LGBFitter()
    _, _, res, _ = fitter.train_k_fold(kfold, df_train, df_test, params)
    res_list[p] = res
    print(f'lr: {p}, res: {res}')
pd.DataFrame(res_list).to_csv(f'param_range/res_{boosting_mode}_{extra_tree}_bagging_fraction.csv', index=False)

res_list = {}
for p in [0.2, 0.4, 0.6, 0.8]:
    params = common_params.copy()
    params['bagging_fraction'] = p
    kfold = KFold(n_splits=5)
    fitter = LGBFitter()
    _, _, res, _ = fitter.train_k_fold(kfold, df_train, df_test, params)
    res_list[p] = res
    print(f'lr: {p}, res: {res}')
pd.DataFrame(res_list).to_csv(f'param_range/res_{boosting_mode}_{extra_tree}_bagging_fraction.csv', index=False)

res_list = {}
for p in [2, 4, 6, 8]:
    params = common_params.copy()
    params['lambda_l1'] = p
    kfold = KFold(n_splits=5)
    fitter = LGBFitter()
    _, _, res, _ = fitter.train_k_fold(kfold, df_train, df_test, params)
    res_list[p] = res
    print(f'lr: {p}, res: {res}')
pd.DataFrame(res_list).to_csv(f'param_range/res_{boosting_mode}_{extra_tree}_lambda_l1.csv', index=False)

res_list = {}
for p in [2, 4, 6, 8]:
    params = common_params.copy()
    params['lambda_l2'] = p
    kfold = KFold(n_splits=5)
    fitter = LGBFitter()
    _, _, res, _ = fitter.train_k_fold(kfold, df_train, df_test, params)
    res_list[p] = res
    print(f'lr: {p}, res: {res}')
pd.DataFrame(res_list).to_csv(f'param_range/res_{boosting_mode}_{extra_tree}_lambda_l2.csv', index=False)