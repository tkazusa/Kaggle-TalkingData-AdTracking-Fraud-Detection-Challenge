# -*- encoding: UTF-8 -*-
import pandas as pd


df_test = pd.read_csv("input/test.csv")
df_test['click_time'] = pd.to_datetime(df_test['click_time'])
df_test.to_pickle("input/test.pkl")


df_test_s = pd.read_csv("input/test_supplement.csv")
df_test_s['click_time'] = pd.to_datetime(df_test_s['click_time'])
df_test_s.to_pickle("input/test_supplement.pkl")


df_train = pd.read_csv("input/train.csv")
df_train['click_time'] = pd.to_datetime(df_train['click_time'])
df_train.to_pickle("input/train.pkl")
