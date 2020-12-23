import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import KFold
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

# load data
train_path = '../toy_data/train_onehot.csv'
test_path = '../toy_data/test_onehot.csv'
df_test = pd.read_csv(test_path)
df_train = pd.read_csv(train_path)

x_train, y_train = df_train.drop(columns=['label']).values, df_train['label'].values
x_test, y_test = df_test.drop(columns=['label']).values, df_test['label'].values

num_round = 20

#0.20397999999999997
gbdt_param = {'bagging_fraction': 0.7744376536407631, 'bagging_freq': 10, 'boosting': 'gbdt',
              'feature_fraction': 0.6527670858085077, 'lambda_l1': 0.012029176681726539,
              'lambda_l2': 4.903602405927458, 'learning_rate': 0.010670743157696252,
              'metric': 'binary_error', 'min_gain_to_split': 0.008743388851820424,
              'num_leaves': 135, 'num_threads': 64, 'objective': 'binary'}

#0.20563000000000003
goss_param = {'num_thread': 64, 'num_leaves': 80, 'metric': 'binary_error', 'objective': 'binary',
              'learning_rate': 0.01, 'feature_fraction': 0.8, 'boosting': 'goss'}

#0.20643000000000003
dart_param = {'num_thread': 64, 'num_leaves': 128, 'metric': 'binary_error', 'objective': 'binary',
              'learning_rate': 0.07, 'feature_fraction': 0.4, 'bagging_fraction': 0.9, 'bagging_freq': 10, 'boosting': 'dart'}

kfold = KFold(n_splits=5)
out_list = []

for train_index, test_index in kfold.split(x_train):

    X_train_, X_test_ = x_train[train_index], x_train[test_index]
    y_train_, y_test_ = y_train[train_index], y_train[test_index]
    train_data = lgb.Dataset(data=X_train_, label=y_train_)

    bst_gbdt = lgb.train(gbdt_param, train_data, num_round)
    bst_goss = lgb.train(goss_param, train_data, num_round)
    bst_dart = lgb.train(dart_param, train_data, num_round)

    gbdt_out = bst_gbdt.predict(x_test)
    goss_out = bst_goss.predict(x_test)
    dart_out = bst_goss.predict(x_test)

    out = pd.DataFrame()
    out['gbdt'] = gbdt_out
    out['goss'] = goss_out
    out['dart'] = dart_out
    out['label'] = y_test_

    out_list.append(out)

out_ = pd.concat(out_list)

# our data
'''
gbdt, goss, dart output
for original data, we fillna with median, then tsne and use knn.
compute max, min, std, mean
'''

#df_train_original = pd.read_csv('../toy_data/train_n.csv')

df_train_filled = df_train.fillna(df_train.median())
x_train_filled = df_train_filled.drop(columns=['label'])
x_train_filled = x_train_filled.dropna(axis=1)

tsne = TSNE(n_components=3)
tsne.fit_transform(x_train_filled)
df_tsne_embedding = pd.DataFrame(tsne.embedding_)

print(out_.shape, df_tsne_embedding.shape)
'''
out_1 = pd.concat([out, df_tsne_embedding, knn_out], axis=1)
'''
#knn = KNeighborsClassifier()


