# -*- encoding: UTF-8 -*-
import gc

import pandas as pd
import numpy as np

from tools.util import Util
from tools.encoding import MeanEncoder, HashEncoder, FreqencyEncoder

from sklearn.model_selection import KFold
np.random.seed(10)


if __name__ == "__main__":
    df_selected_train = pd.read_pickle("input/org/selected_train.pkl")
    df_selected_val = pd.read_pickle("input/org/selected_val.pkl")
    df_org_test = pd.read_pickle("input/org/org_test.pkl")
    kf = KFold(n_splits=3, shuffle=False)
    columns_list = df_selected_train.columns.drop(["click_id", "is_attributed", "data_set", "day", "hour"])

    for columns in columns_list:
        print(columns)
        dfs = []
        for tr_ind, val_ind in kf.split(df_selected_train[columns]):
            X_tr, X_val = df_selected_train[columns].iloc[tr_ind], df_selected_train[columns].iloc[val_ind]
            y_tr, y_val= df_selected_train['is_attributed'].iloc[tr_ind], df_selected_train['is_attributed'].iloc[val_ind]

            me = MeanEncoder()
            me.fit(X_tr, y_tr)
            me.transform(X_val)
            dfs.append(me.transform(X_val))

        df_selected_train["ME_" + columns] = pd.concat(dfs, axis=0)
        df_selected_val["ME_" + columns] = me.transform(df_selected_val[columns])
        df_org_test["ME_" + columns] = me.transform(df_org_test[columns])

    df_train = df_selected_train.drop(columns_list, axis=1)
    df_train = df_train.drop(["click_id", "data_set", "is_attributed"], axis=1)

    df_val = df_selected_val.drop(columns_list, axis=1)
    df_val = df_val.drop(["click_id", "data_set", "is_attributed"], axis=1)

    df_test = df_org_test.drop(columns_list, axis=1)
    df_test = df_test.drop(["click_id", "data_set", "is_attributed"], axis=1)

    df_test["click_id"] = df_org_test["click_id"].astype('int')
    df_test["hour"] = df_org_test["hour"].astype('int')
    df_train.to_pickle("input/org/MeanEncoded/test.pkl")

    df_train["hour"] = df_selected_train["hour"].astype('int')
    df_train.to_pickle("input/org/MeanEncoded/train.pkl")

    df_val["hour"] = df_selected_val["hour"]
    df_train.to_pickle("input/org/MeanEncoded/val.pkl")
