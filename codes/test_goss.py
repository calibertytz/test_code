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

'''
num_leaves = 32
num_round = 2000
learning_rate = 3e-2
feature_fraction = 0.8
boosting_mode = 'goss'
extra_trees = True


for feature_fraction in tqdm([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]):
    common_params = {'num_thread': 64,
                     'num_leaves': num_leaves,
                     'metric': 'binary_error',
                     'objective': 'binary',
                     'num_round': num_round,
                     'learning_rate': learning_rate,
                     'feature_fraction': feature_fraction,
                     'boosting': boosting_mode,
                     'extra_trees': extra_trees
                     }

    model_fitter = LGBFitter(label='label')
    kfold = KFold(n_splits=5)

    _,_,acc,_ = model_fitter.train_k_fold(kfold, df_train, df_test, params=common_params)
    print(common_params)
    print(np.mean(acc))
'''


@dataclass
class LGBOpt:
    num_threads: any = hp.choice('num_threads', [64])
    num_leaves: any = hp.choice('num_leaves', [78, 79, 80, 81, 82])
    metric: any = hp.choice('metric', ['binary_error'])
    num_round: any = hp.choice('num_rounds', [2000])
    objective: any = hp.choice('objective', ['binary'])
    learning_rate: any = hp.uniform('learning_rate', 0.01, 0.05)
    feature_fraction: any = hp.uniform('feature_fraction', 0.75, 0.85)

    boosting: any = hp.choice('boosting', ['goss'])
    lambda_l1: any = hp.uniform('lambda_l1', 0, 10)
    lambda_l2: any = hp.uniform('lambda_l2', 0, 10)
    min_gain_to_split: any = hp.uniform('min_gain_to_split', 0, 1)
    min_data_in_bin = hp.choice('min_data_in_bin', [3, 5, 10, 15, 20, 50])


# search_k_fold
lgb_opt = LGBOpt()
model_fitter = LGBFitter(label='label', opt=lgb_opt)
kfold = KFold(n_splits=5)
model_fitter.search_k_fold(k_fold=kfold, data=df_train)
print(model_fitter.opt_params)
