import open_competition as op
from open_competition.tabular.model_fitter import *
import pandas as pd

# read data
train_path = '../toydata/train_n.csv'
test_path = '../toydata/test_n.csv'
df_test = pd.read_csv(test_path)
df_train = pd.read_csv(train_path)

lgb_fitter = LGBFitter()


# test
print(lgb_fitter.train(df_train, df_test))