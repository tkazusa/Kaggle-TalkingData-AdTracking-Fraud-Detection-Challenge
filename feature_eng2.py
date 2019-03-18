# -* encoding: UTF-8 -*-
import gc
from pathlib import Path

import numpy as np
import pandas as pd

DATADIR = Path(
    '~/work/TalkingDataAdTrackingFraudDetectionChallenge/data/input')
tr_path = DATADIR / 'train_small.csv'
test_path = DATADIR / 'test_small.csv'

train_cols = ['ip', 'app', 'device', 'os',
              'channel', 'click_time', 'is_attributed']
test_cols = ['ip', 'app', 'device', 'os', 'channel', 'click_time']


def load_datasets():
    train = pd.read_csv(tr_path, usecols=train_cols)
    print(train.head())
    test = pd.read_csv(test_path, usecols=test_cols)
    print(test.head())
    return train, test


def bind_tr_test(train, test):
    train, test = load_datasets()
    len_train = len(train)
    print('The initial size of the train set is', len_train)
    print('Binding the training and test set together...')
    data = train.append(test)

    del test
    gc.collect()

    return data


def create_time_features(data):
    print("Creating new time features: 'hour' and 'day'...")
    data['hour'] = pd.to_datetime(data.click_time).dt.hour.astype('uint8')
    data['day'] = pd.to_datetime(data.click_time).dt.day.astype('uint8')

    gc.collect()
    return data


def create_count_channels_features(data):
    print("Creating new count features: 'n_channels', 'ip_app_count', 'ip_app_os_count'...")
    print('Computing the number of channels associated with ')
    print('a given IP address within each hour...')
    print('一時間の中でIPアドレス毎のチャネル数を数えている')
    n_chans = train[['ip', 'day', 'hour', 'channel']].groupby(by=['ip', 'day', 'hour'])[
        ['channel']].count().reset_index().rename(columns={'channel': 'n_channels'})

    print('Merging the channels data with the main data set...')
    data = data.merge(n_chans, on=['ip', 'day', 'hour'], how='left')
    del n_chans
    gc.collect()

    print('Computing the number of channels associated with ')
    print('a given IP address and app...')
    print('IPアドレス毎/app毎のチャネル数を数えている')
    n_chans = train[['ip', 'app', 'channel']].groupby(by=['ip', 'app'])[
        ['channel']].count().reset_index().rename(columns={'channel': 'ip_app_count'})

    print('Merging the channels data with the main data set...')
    data = data.merge(n_chans, on=['ip', 'app'], how='left')
    del n_chans
    gc.collect()

    print('Computing the number of channels associated with ')
    print('a given IP address, app, and os...')
    print('IPアドレス毎/app毎/os毎のチャネル数を数えている')
    n_chans = train[['ip', 'app', 'os', 'channel']].groupby(by=['ip', 'app', 'os'])[
        ['channel']].count().reset_index().rename(columns={'channel': 'ip_app_os_count'})

    print('Merging the channels data with the main data set...')
    data = data.merge(n_chans, on=['ip', 'app', 'os'], how='left')
    del n_chans
    gc.collect()

    print("Adjusting the data types of the new count features... ")
    data.info()
    data['n_channels'] = data['n_channels'].astype('uint16')
    data['ip_app_count'] = data['ip_app_count'].astype('uint16')
    data['ip_app_os_count'] = data['ip_app_os_count'].astype('uint16')
    data['is_attributed'] = data['is_attributed'].astype('category')

    return data


def train_test_split(data, len_train):
    train = data[:len_train]
    test = data[len_train:]
    return train, test


def negative_down_sampling(data, random_state, target_variable):
    positive_data = data[data[target_variable] == 1]
    positive_ratio = float(len(positive_data)) / len(data)
    negative_data = data[data[target_variable] == 0].sample(
        frac=positive_ratio / (1 - positive_ratio), random_state=random_state)
    return pd.concat([positive_data, negative_data])


if __name__ == "__main__":
    train, test = load_datasets()
    len_train = len(train)
    data = bind_tr_test(train, test)
    data = create_time_features(data)
    data = create_count_channels_features(data)

    del data
    gc.collect()

    train, test = train_test_split(data, len_train)
    sampled_train = negative_down_sampling(
        train, target_variable='is_attributed', random_state=3655)

    del train
    gc.collect()

    print(sampled_train.head())
    print(test.head())
