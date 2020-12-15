import open_competition as op
from open_competition.tabular.model_fitter import *
import pandas as pd
from sklearn.model_selection import KFold

# load data
train_path = '../toy_data/train_onehot.csv'
test_path = '../toy_data/test_onehot.csv'
df_test = pd.read_csv(test_path)
df_train = pd.read_csv(train_path)


lgb_fitter = LGBFitter(max_eval=1)


if __name__ == '__main__':
    print('searching! \n')
    lgb_fitter.search(df_train, df_test)
    print(lgb_fitter.opt_params)

    print('kfold searching! \n')
    kfold = KFold(n_splits=2)
    lgb_fitter.search_k_fold(kfold, df_train)
    print(lgb_fitter.opt_params)