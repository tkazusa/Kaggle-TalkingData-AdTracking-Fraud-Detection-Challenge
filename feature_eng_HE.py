# -*- encoding: UTF-8 -*-
import gc

import pandas as pd
import numpy as np

from tools.util import Util
from tools.encoding import MeanEncoder, HashEncoder, FreqencyEncoder

if __name__ == "__main__":
    print("start")
    df_selected_train = pd.read_pickle("input/org/selected_train.pkl")
    df_selected_val = pd.read_pickle("input/org/selected_val.pkl")
    df_org_test = pd.read_pickle("input/org/org_test.pkl")

    columns_list = df_selected_train.columns.drop(["click_id", "is_attributed", "data_set", "day", "hour"])

    df_train = pd.DataFrame()
    df_val   = pd.DataFrame()
    df_test  = pd.DataFrame()

    for columns in columns_list:
        print(columns)
        he = HashEncoder(n_features=20)
        he.fit(df_selected_train[columns])
        
        df_tmp_train = he.transform(df_selected_train[columns])
        df_train[df_tmp_train.columns] = df_tmp_train
        del df_tmp_train 
        gc.collect()
        
        
        df_tmp_val = he.transform(df_selected_val[columns])
        df_val[df_tmp_val.columns] = df_tmp_val
        del df_tmp_val
        gc.collect()
     
        df_tmp_test = he.transform(df_org_test[columns])
        df_test[df_tmp_test.columns] = df_tmp_test
        del df_tmp_test
        gc.collect()
        
        
    df_test["click_id"] = df_org_test["click_id"].astype('int')
    df_test["hour"] = df_org_test["hour"].astype('int')
    df_train.to_pickle("input/org/HashEncoded/test.pkl")

    df_train["hour"] = df_selected_train["hour"].astype('int')
    df_train.to_pickle("input/org/HashEncoded/train.pkl")

    df_val["hour"] = df_selected_val["hour"]
    df_train.to_pickle("input/org/HashEncoded/val.pkl")
