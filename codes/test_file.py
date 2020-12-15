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

# load data
train_path = '../toy_data/train_onehot.csv'
test_path = '../toy_data/test_onehot.csv'
df_test = pd.read_csv(test_path)
df_train = pd.read_csv(train_path)

cpu_count = 4

'''
need to test range for the following param:
1. num_leaves
2. num_rounds
3. learning_rate
4. boosting
5. 

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
        return {'num_thread': 4, 'num_leaves': 12, 'metric': 'binary', 'objective': 'binary',
                'num_round': 1000, 'learning_rate': 0.01, 'feature_fraction': 0.8, 'bagging_fraction': 0.8}

common_params = LGBOpt.get_common_params()
fitter = LGBFitter()
kfold = KFold(n_splits=5)

# num_leaves
print('find num leaves \n')
for i in range(12, 64+1):
    param = common_params
    common_params['num_leaves'] = i
    _, _, res, _ = fitter.train(df_train, df_test, params=param)
    print(f'num leaves:{i}, acc: {res}')

# learning_rate
print('find learning_rate \n')
for i in range(1, 11):
    param = common_params
    common_params['learning_rate'] = i/10
    _, _, res, _ = fitter.train(df_train, df_test, params=param)
    print(f'learning_rate:{i/10}, acc: {res}')

# feature_fraction
print('find feature_fraction \n')
for i in np.linspace(0.5, 1, 10):
    param = common_params
    common_params['feature_fraction'] = i
    _, _, res, _ = fitter.train(df_train, df_test, params=param)
    print(f'feature_fraction:{i}, acc: {res}')

# bagging_fraction
print('find bagging_fraction \n')
for i in np.linspace(0.5, 1, 10):
    param = common_params
    common_params['bagging_fraction'] = i
    _, _, res, _ = fitter.train(df_train, df_test, params=param)
    print(f'bagging_fraction:{i}, acc: {res}')

# boosting
print('find boosting \n')
for x in ['gbdt', 'dart', 'goss']:
    param = common_params
    common_params['boosting'] = x
    _, _, res, _ = fitter.train(df_train, df_test, params=param)
    print(f'boosting:{x}, acc: {res}')

