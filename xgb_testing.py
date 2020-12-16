# encoding:utf-8
import pandas as pd
from open_competition.tabular.encoder import CategoryEncoder
from open_competition.tabular.model_fitter import XGBFitter
from sklearn.model_selection import KFold


def one_hot_transform(df):
    one_hot_encoder = CategoryEncoder()
    df_ = df.copy(deep=True)
    cols = list(df_.columns)
    discrete_cols = [x for x in cols if x[0] == 'd']
    one_hot_encoder.fit(df=df_, y='label', targets=discrete_cols, configurations=[('one-hot', None)])
    transformed_df = one_hot_encoder.transform(df_)
    return transformed_df.drop(columns=discrete_cols)


# read data
train_path = '../toy_data/train_n.csv'
test_path = '../toy_data/test_n.csv'
df_test = pd.read_csv(test_path)
df_train = pd.read_csv(train_path)

df_train = df_train.drop(columns=['company_id', 'obs_date'])
df_test = df_test.drop(columns=['company_id', 'obs_date'])

df_train = one_hot_transform(df_train)
df_test = one_hot_transform(df_test)

df_train.to_csv('../toy_data/train_onehot.csv', index=False)
df_test.to_csv('../toy_data/test_onehot.csv', index=False)

#
# @dataclass
# class XGBOpt:
#     nthread: any = hp.choice('nthread', [cpu_count])
#     eval_metric: any = hp.choice('eval_metric', ['error'])
#     booster: any = hp.choice('booster', ['gbtree', 'dart'])
#     sample_type: any = hp.choice('sample_type', ['uniform', 'weighted'])
#     rate_drop: any = hp.uniform('rate_drop', 0, 0.2)
#     objective: any = hp.choice('objective', ['binary:logistic'])
#     max_depth: any = hp.choice('max_depth', [4, 5, 6, 7, 8])
#     num_round: any = hp.choice('num_round', [100])
#     eta: any = hp.uniform('eta', 0.01, 0.1)
#     subsample: any = hp.uniform('subsample', 0.8, 1)
#     colsample_bytree: any = hp.uniform('colsample_bytree', 0.3, 1)
#     gamma: any = hp.choice('gamma', [0, 1, 5])
#     min_child_weight: any = hp.uniform('min_child_weight', 0, 15)
#     sampling_method: any = hp.choice('sampling_method', ['uniform', 'gradient_based'])
#     @staticmethod
#     def get_common_params():
#         return {'nthread': 4, 'max_depth': 3, 'eval_metric': 'error', 'object': 'binary:logistic', 'eta': 0.01,
#                 'subsample': 0.8, 'colsample_bytree': 0.8, 'num_round': 1000}
#

# not change
booster = "gbtree"
eval_metric = 'error'
objective = 'binary:logistic'
cpu_count = 32

# need_change
num_round = 2000

# need
gamma = 0
max_depth = 6
min_child_weight = 1
subsample = 1
sampling_method = "uniform"
colsample_bytree = 0.5
eta = 1e-2

common_params = {'nthread': cpu_count, 'max_depth': max_depth, 'eval_metric': eval_metric, 'objective': objective,
                 'eta': eta,
                 'subsample': subsample, "sampling_method": sampling_method, 'colsample_bytree': colsample_bytree,
                 'num_round': num_round}


def run_model(tune_param, tune_value):
    params = common_params.copy()
    params[tune_param] = tune_value
    kfold = KFold(n_splits=5)
    fitter = XGBFitter()
    _, _, res, _ = fitter.train_k_fold(kfold, df_train, df_test, params)
    print(f'tune_param is {tune_param},tune_value is {tune_value} ,result is:\n{res}')


for learning_rate in [1e-2, 2e-2, 3e-2, 5e-2]:
    run_model("eta", learning_rate)
