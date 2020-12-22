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


#num_leaves = 1024
num_round = 2000
learning_rate = 2e-2
feature_fraction = 0.8
bagging_fraction = 0.8
bagging_freq = 20
boosting_mode = 'rf'
max_depth = 32


for max_depth in tqdm([32]):
    common_params = {'num_thread': 64,
                     'max_depth': max_depth,
                     'metric': 'binary_error',
                     'objective': 'binary',
                     'num_round': num_round,
                     'learning_rate': learning_rate,
                     'feature_fraction': feature_fraction,
                     'bagging_fraction': bagging_fraction,
                     'bagging_freq': bagging_freq,
                     'boosting': boosting_mode
                     }

    model_fitter = LGBFitter(label='label')
    kfold = KFold(n_splits=5)

    _,_,acc,_ = model_fitter.train_k_fold(kfold, df_train, df_test, params=common_params)
    print(common_params)
    print(np.mean(acc))