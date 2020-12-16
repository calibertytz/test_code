import open_competition as op
from open_competition.tabular.model_fitter import LRFitter
from open_competition.tabular.encoder import CategoryEncoder
import pandas as pd
from ind_cols import get_ind_col

from dataclasses import dataclass, asdict
from sklearn.model_selection import KFold
from dataclasses import dataclass, asdict
import hyperopt.pyll
from hyperopt import fmin, tpe, hp
import copy
import numpy as np
from tqdm import tqdm
import sys

def target_mean(df_train, df_test):
    df_train = df_train.drop(columns=['company_id', 'obs_date'])
    df_test = df_test.drop(columns=['company_id', 'obs_date'])

    encoder = CategoryEncoder()
    cols = list(df_train.columns)
    discrete_cols = [x for x in cols if x[0] == 'd']
    encoder.fit(df=df_train, y='label', targets=discrete_cols, configurations=[('target', {'smoothing': 0.5})])

    transformed_df_train = encoder.transform(df_train, y='label')
    transformed_df_test = encoder.transform(df_test)

    df_1, df_2 = transformed_df_train.drop(columns = discrete_cols), transformed_df_test.drop(columns=discrete_cols)
    cols_ = df_1.columns
    for col in cols_:
        df_1[col] = df_1[col].fillna(df_1[col].mean())
        df_2[col] = df_2[col].fillna(df_2[col].mean())
    ind_cols = get_ind_col(df_1)
    return df_1[ind_cols], df_2[ind_cols]

# load data
train_path = 'toy_data/train_n.csv'
test_path = 'toy_data/test_n.csv'
df_test = pd.read_csv(test_path)
df_train = pd.read_csv(train_path)

df_train, df_test = target_mean(df_train, df_test)

print(df_train)

fiter = LRFitter()
kfold = KFold(n_splits=5)
#fiter.train_k_fold(kfold, df_train, df_test)
fiter.search_k_fold(kfold, df_train)
