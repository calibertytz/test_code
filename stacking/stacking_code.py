import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import KFold


# load data
train_path = '../toy_data/train_onehot.csv'
test_path = '../toy_data/test_onehot.csv'
df_test = pd.read_csv(test_path)
df_train = pd.read_csv(train_path)
x_train, y_train = df_train.drop(columns=['label']), df_train['label']
x_test, y_test = df_test.drop(columns=['label']), df_test['label']



train_data = lgb.Dataset(data=x_train, label=y_train)
test_data = lgb.Dataset(data=x_test)

num_round = 2000

#0.20397999999999997
gbdt_param = {'bagging_fraction': 0.7744376536407631, 'bagging_freq': 0, 'boosting': 0,
              'feature_fraction': 0.6527670858085077, 'lambda_l1': 0.012029176681726539,
              'lambda_l2': 4.903602405927458, 'learning_rate': 0.010670743157696252,
              'metric': 'binary_error','min_gain_to_split': 0.008743388851820424,
              'num_leaves': 135, 'num_threads': 64, 'objective': 'binary'}

#0.20563000000000003
goss_param = {'num_thread': 64, 'num_leaves': 80, 'metric': 'binary_error', 'objective': 'binary',
              'learning_rate': 0.01, 'feature_fraction': 0.8, 'boosting': 'goss'}

#0.20643000000000003
dart_param = {'num_thread': 64, 'num_leaves': 128, 'metric': 'binary_error', 'objective': 'binary',
              'learning_rate': 0.07, 'feature_fraction': 0.4, 'bagging_fraction': 0.9, 'bagging_freq': 10, 'boosting': 'dart'}

bst_gbdt = lgb.train(gbdt_param, train_data, num_round)
bst_goss = lgb.train(goss_param, train_data, num_round)
bst_dart = lgb.train(dart_param, train_data, num_round)

gbdt_out = bst_gbdt.predict(test_data).values
goss_out = bst_goss.predict(test_data).values
dart_out = bst_goss.predict(test_data).values

out = pd.DataFrame()
out['gbdt'] = gbdt_out
out['goss'] = goss_out
out['dart'] = dart_out
out['label'] = y_test

out.to_csv('lgb_out.csv')

