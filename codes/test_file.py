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
        return {'num_thread': 8, 'num_leaves': 12, 'metric': 'binary', 'objective': 'binary',
                'num_round': 1000, 'learning_rate': 0.01, 'feature_fraction': 0.8, 'bagging_fraction': 0.8}


fitter = LGBFitter()
kfold = KFold(n_splits=5)

boosting_mode = sys.argv[1] # 'gbdt', 'dart', 'goss'
common_params = LGBOpt.get_common_params()
common_params['boosting'] = boosting_mode

# num_round
print('find num_round \n')
res_num_round = {}
for i in tqdm(np.arange(1500, 3000, 100)):
    param = common_params.copy()
    param['num_round'] = i
    _, _, res, _ = fitter.train_k_fold(kfold, df_train, df_test, params=param)
    res_num_round[i] = res
    print(f'num_round:{i}, acc: {res}')
pd.DataFrame(res_num_round).to_csv(f'param_range/num_round_{boosting_mode}.csv')

# num_leaves
print('find num leaves \n')
res_num_leaves = {}
for i in tqdm(range(12, 64+1)):
    param = common_params.copy()
    param['num_leaves'] = i
    _, _, res, _ = fitter.train_k_fold(kfold, df_train, df_test, params=param)
    print(f'num leaves:{i}, acc: {res}')
    res_num_leaves[i] = res
pd.DataFrame(res_num_leaves).to_csv(f'param_range/num_leaves_{boosting_mode}.csv')

# learning_rate
print('find learning_rate \n')
res_learning_rate = {}
for i in tqdm(range(1, 11)):
    param = common_params.copy()
    param['learning_rate'] = i/10
    _, _, res, _ = fitter.train_k_fold(kfold, df_train, df_test, params=param)
    res_learning_rate[i] = res
    print(f'learning_rate:{i/10}, acc: {res}')
pd.DataFrame(res_learning_rate).to_csv(f'param_range/learning_rate_{boosting_mode}.csv')


# feature_fraction
print('find feature_fraction \n')
res_feature_fraction = {}
for i in tqdm(np.linspace(0.5, 1, 10)):
    param = common_params.copy()
    param['feature_fraction'] = i
    _, _, res, _ = fitter.train_k_fold(kfold, df_train, df_test, params=param)
    res_feature_fraction[i] = res
    print(f'feature_fraction:{i}, acc: {res}')
pd.DataFrame(res_num_round).to_csv(f'param_range/num_round_{boosting_mode}.csv')


# bagging_fraction
print('find bagging_fraction \n')
res_bagging_fraction = {}
for i in tqdm(np.linspace(0.5, 1, 10)):
    param = common_params.copy()
    param['bagging_fraction'] = i
    _, _, res, _ = fitter.train_k_fold(kfold, df_train, df_test, params=param)
    res_bagging_fraction[i] = res
    print(f'bagging_fraction:{i}, acc: {res}')
pd.DataFrame(res_num_round).to_csv(f'param_range/num_round_{boosting_mode}.csv')

'''
then use these range to search.
'''