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

then for num_leaves, 

for gbdt, max is 96 fix as 32

for dart, 

for goss, if num_leaves is large, the needed num_round is small. max is 64 fix as 32



'''


@dataclass
class LGBOpt:
    num_threads: any = hp.choice('num_threads', [cpu_count])
    num_leaves: any = hp.choice('num_leaves', [64])
    metric: any = hp.choice('metric', ['binary_error'])
    num_round: any = hp.choice('num_rounds', [1000])
    objective: any = hp.choice('objective', ['binary'])
    learning_rate: any = hp.uniform('learning_rate', 0.01, 0.1)
    feature_fraction: any = hp.uniform('feature_fraction', 0.5, 1.0)
    bagging_fraction: any = hp.uniform('bagging_fraction', 0.8, 1.0)
    boosting: any = hp.choice('boosting', ['gbdt', 'dart', 'goss'])
    extra_trees: any = hp.choice('extra_tress', [False, True])
    drop_rate: any = hp.uniform('drop_rate', 0, 0.2)
    uniform_drop: any = hp.choice('uniform_drop', [True, False])
    lambda_l1: any = hp.uniform('lambda_l1', 0, 10)  # TODO: Check range
    lambda_l2: any = hp.uniform('lambda_l2', 0, 10)  # TODO: Check range
    min_gain_to_split: any = hp.uniform('min_gain_to_split', 0, 1)  # TODO: Check range
    min_data_in_bin = hp.choice('min_data_in_bin', [3, 5, 10, 15, 20, 50])

    @staticmethod
    def get_common_params():
        return {'num_thread': 8, 'num_leaves': 12, 'metric': 'binary_error', 'objective': 'binary',
                'num_round': 1000, 'learning_rate': 0.01, 'feature_fraction': 0.8, 'bagging_fraction': 0.8}


num_leaves = 32
num_round =2000
learning_rate = 5e-2
boosting_mode = sys.argv[1] # 'gbdt', 'dart', 'goss'

common_params = {'num_thread': 8, 'num_leaves': num_leaves, 'metric': 'binary_error', 'objective': 'binary',
                'num_round': num_round, 'learning_rate': learning_rate, 'feature_fraction': 0.8, 'bagging_fraction': 0.8}

common_params['boosting'] = boosting_mode

'''
for num_round in tqdm([1000, 1500, 2000]):
    params = common_params.copy()
    params['num_round'] = num_round
    print(params)
    kfold = KFold(n_splits=5)
    fitter = LGBFitter()
    _, _, res, _ = fitter.train_k_fold(kfold, df_train, df_test, params)
    print(f'{num_round}, {res}')
'''

'''
for learning_rate in tqdm([2e-2, 3e-2, 4e-2]):
    params = common_params.copy()
    params['learning_rate'] = learning_rate
    print(params)
    kfold = KFold(n_splits=5)
    fitter = LGBFitter()
    _, _, res, _ = fitter.train_k_fold(kfold, df_train, df_test, params)
    print(f'{learning_rate}, {res}')
'''

'''
for num_leaves in tqdm([16, 32, 48, 64, 72, 96]):
    params = common_params.copy()
    params['num_leaves'] = num_leaves
    print(params)
    kfold = KFold(n_splits=5)
    fitter = LGBFitter()
    _, _, res, _ = fitter.train_k_fold(kfold, df_train, df_test, params)
    print(f'{num_leaves}, {res}')
'''


for feature_fraction in tqdm([0.2, 0.4, 0.6, 0.8]):
    params = common_params.copy()
    params['feature_fraction'] = feature_fraction
    print(params)
    kfold = KFold(n_splits=5)
    fitter = LGBFitter()
    _, _, res, _ = fitter.train_k_fold(kfold, df_train, df_test, params)
    print(f'{feature_fraction}, {res}')


'''
for num_leaves in [16, 32, 64, 96]:
    params = common_params.copy()
    params['']
    print(params)
    fitter = LGBFitter()

fiter_option = LGBOpt()
'''