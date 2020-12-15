import open_competition as op
from open_competition.tabular.model_fitter import *
from open_competition.tabular.encoder import CategoryEncoder
import pandas as pd


def one_hot_transform(df):
    one_hot_encoder = CategoryEncoder()
    df_ = df.copy(deep=True)
    cols = list(df_.columns)
    discrete_cols = [x for x in cols if x[0] == 'd']
    one_hot_encoder.fit(df=df_, y='label', targets=discrete_cols, configurations=('one_hot', None))
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

lgb_fitter = LGBFitter()


# test
params = {'num_thread': 4, 'num_leaves': 12, 'metric': 'binary', 'objective': 'binary',
                'num_round': 100, 'learning_rate': 0.01, 'feature_fraction': 0.8, 'bagging_fraction': 0.8}
print(lgb_fitter.train(df_train, df_test, params))