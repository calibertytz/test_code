import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import KFold
import numpy as np
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

# load data
train_path = '../toy_data/train_onehot.csv'
test_path = '../toy_data/test_onehot.csv'
df_test = pd.read_csv(test_path)
df_train = pd.read_csv(train_path)

x_train, y_train = df_train.drop(columns=['label']).values, df_train['label'].values
x_test, y_test = df_test.drop(columns=['label']).values, df_test['label'].values

'''
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
test_out_list = []
for train_index, test_index in tqdm(kfold.split(x_train)):

    X_train_, X_test_ = x_train[train_index], x_train[test_index]
    y_train_, y_test_ = y_train[train_index], y_train[test_index]
    X_test__, y_test__ = x_test[test_index], y_test[test_index]

    train_data = lgb.Dataset(data=X_train_, label=y_train_)

    bst_gbdt = lgb.train(gbdt_param, train_data, num_round)
    bst_goss = lgb.train(goss_param, train_data, num_round)
    bst_dart = lgb.train(dart_param, train_data, num_round)

    gbdt_out = bst_gbdt.predict(X_test_)
    goss_out = bst_goss.predict(X_test_)
    dart_out = bst_goss.predict(X_test_)

    gbdt_out_test = bst_gbdt.predict(X_test__)
    goss_out_test = bst_goss.predict(X_test__)
    dart_out_test = bst_goss.predict(X_test__)

    out = pd.DataFrame()
    out['gbdt'] = gbdt_out
    out['goss'] = goss_out
    out['dart'] = dart_out
    out['label'] = y_test_

    out_test = pd.DataFrame()
    out_test['gbdt'] = gbdt_out_test
    out_test['goss'] = goss_out_test
    out_test['dart'] = dart_out_test
    out_test['label'] = y_test__

    out_list.append(out)
    test_out_list.append(out_test)

out_ = pd.concat(out_list)
out_test = pd.concat(test_out_list)

out_.to_csv('out_.csv', index=False)
out_test.to_csv('out_test.csv', index=False)

print('model done!')
'''

out_ = pd.read_csv('out_.csv')
out_test = pd.read_csv('out_test.csv')


# our data
'''
gbdt, goss, dart output
for original data, we fillna with median, then tsne and use knn.
compute max, min, std, mean
'''

#df_train_original = pd.read_csv('../toy_data/train_n.csv')
'''
df_train_filled = df_train.fillna(df_train.median())
x_train_filled = df_train_filled.drop(columns=['label'])
x_train_filled = x_train_filled.dropna(axis=1)

df_test_filled = df_test.fillna(df_test.median())
x_test_filled = df_test_filled.drop(columns=['label'])
x_test_filled = x_test_filled.dropna(axis=1)
'''
# tsne
stacked_data = pd.concat([df_train, df_test])
stacked_data.fillna(stacked_data.median(), inplace=True)
stacked_data.drop(columns=['label'], inplace=True)
stacked_data.dropna(axis=1)


tsne = TSNE(n_components=3, n_jobs=-1)
tsne.fit(stacked_data)
df_embedding = pd.DataFrame(tsne.embedding_)
df_tsne_train = df_embedding.iloc[:100000, :]
df_tsne_test = df_embedding.iloc[100000:, :]

df_tsne_train.to_csv('tsne_result_train.csv', index=False)
df_tsne_test.to_csv('tsne_result_test.csv', index=False)

print('tsne done!')

# knn
# train
neigh = NearestNeighbors(n_neighbors=5)
neigh.fit(df_tsne_train.values)
knn_index_train = neigh.kneighbors(df_tsne_train.values, 5, return_distance=False)
knn_result_train = []

for index in knn_index_train:
    temp = df_tsne_train.iloc[index, :].mean(axis=1).values
    knn_result_train.append(temp)
df_knn_result_train = pd.DataFrame(np.array(knn_result_train))

out_train_ = pd.concat([df_tsne_train, df_knn_result_train], axis=1, ignore_index=True)
out_train = pd.concat([out_, df_tsne_train, df_knn_result_train], axis=1, ignore_index=True)
# test
neigh = NearestNeighbors(n_neighbors=5)
neigh.fit(df_tsne_test.values)
knn_index_test = neigh.kneighbors(df_tsne_test.values, 5, return_distance=False)
knn_result_test = []

for index in knn_index_test:
    temp = df_tsne_test.iloc[index, :].mean(axis=1).values
    knn_result_test.append(temp)
df_knn_result_test = pd.DataFrame(np.array(knn_result_test))

out_test_ = pd.concat([df_tsne_test, df_knn_result_test], axis=1, ignore_index=True)
out_test = pd.concat([out_test, df_tsne_test, df_knn_result_test], axis=1, ignore_index=True)

# mean min max std

out_train['std'] = out_train_.std(axis=1)
out_train['mean'] = out_train_.mean(axis=1)
out_train['min'] = out_train_.min(axis=1)
out_train['max'] = out_train_.max(axis=1)

out_train.to_csv('out_train.csv', index=False)

out_test['std'] = out_test_.std(axis=1)
out_test['mean'] = out_test_.mean(axis=1)
out_test['min'] = out_test_.min(axis=1)
out_test['max'] = out_test_.max(axis=1)

out_test.to_csv('out_test.csv', index=False)






