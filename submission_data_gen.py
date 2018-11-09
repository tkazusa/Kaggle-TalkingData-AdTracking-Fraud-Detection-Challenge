# -*- encoding: UTF-8 -*-
import gc

import pandas as pd
import numpy as np

from tools.util import Util
from tools.encoding import MeanEncoder, HashEncoder, FreqencyEncoder


if __name__ == "__main__":
    df = pd.read_pickle("input/tr_test_concat.pkl")
    df["day"] = df["click_time"].dt.day.astype('int')
    df["hour"] = df["click_time"].dt.hour.astype('int')
    df = df.drop(["attributed_time"], axis=1)
    df = df.drop(["click_time"], axis=1)
    columns_list = df.columns.drop(["click_id", "is_attributed", "data_set", "day", "hour"])

    df_selected_train = df[(df["data_set"] == 0) &
                           (df["day"] == 8)]
                           
    df_selected_val = df[(df["data_set"] == 0) &
                         (df["day"] == 9) &
                         (df["hour"] >= 4) &
                         (df["hour"] <= 15)]

    df_selected_all = pd.concat([df_selected_train, df_selected_val], axis=0)
    df_org_test = df[df["data_set"] == 1]

    del df
    gc.collect()

    print("saving data")
    df_selected_train.to_pickle("input/org/selected_train.pkl")
    df_selected_val.to_pickle("input/org/selected_val.pkl")
    df_org_test.to_pickle("input/org/org_test.pkl")
