import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import KFold
import numpy as np
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors

# load data
train_path = '../toy_data/train_onehot.csv'
test_path = '../toy_data/test_onehot.csv'
df_test = pd.read_csv(test_path)
df_train = pd.read_csv(train_path)

x_train, y_train = df_train.drop(columns=['label']).values, df_train['label'].values
x_test, y_test = df_test.drop(columns=['label']).values, df_test['label'].values

num_round = 2000

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

    gbdt_out = bst_gbdt.predict(X_test_)
    goss_out = bst_goss.predict(X_test_)
    dart_out = bst_goss.predict(X_test_)

    out = pd.DataFrame()
    out['gbdt'] = gbdt_out
    out['goss'] = goss_out
    out['dart'] = dart_out
    out['label'] = y_test_

    out_list.append(out)

out_ = pd.concat(out_list)

out_.to_csv('out_.csv', index=False)

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

# tsne

tsne = TSNE(n_components=5, n_jobs=-1)
tsne.fit_transform(x_train_filled)
df_tsne_embedding = pd.DataFrame(tsne.embedding_)

df_tsne_embedding.to_csv('tsne_result.csv', index=False)

# knn
neigh = NearestNeighbors(n_neighbors=5)
neigh.fit(df_tsne_embedding.values)
knn_index = neigh.kneighbors(df_tsne_embedding.values, 5, return_distance=False)
knn_result = []

for index in knn_index:
    temp = df_tsne_embedding.iloc[index, :].mean().values
    knn_result.append(temp)
df_knn_result = pd.DataFrame(np.array(knn_result))

out = pd.concat([out_, df_tsne_embedding, df_knn_result], axis=1, ignore_index=True)
out.to_csv('out.csv', index=False)






