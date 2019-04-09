# -* encoding: UTF-8 -*-
import gc
import itertools
import time
import lightgbm as lgb
from pathlib import Path

import numpy as np
import pandas as pd
from gensim import corpora, models

DATADIR = Path(
    '~/work/TalkingDataAdTrackingFraudDetectionChallenge/data')

INPUTDIR = DATADIR / 'input'
WORKDIR = DATADIR / 'work'

tr_path = INPUTDIR / 'train_small.csv'
test_path = INPUTDIR / 'test_small.csv'

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
    data = train.append(test, ignore_index=True, sort=False)

    del train, test
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
    n_chans = data[['ip', 'day', 'hour', 'channel']].groupby(by=['ip', 'day', 'hour'])[
        ['channel']].count().reset_index().rename(columns={'channel': 'n_channels'})
    print('Merging the channels data with the main data set...')
    data = data.merge(n_chans, on=['ip', 'day', 'hour'], how='left')
    del n_chans
    gc.collect()
    data['n_channels'].astype('uint16').to_csv(WORKDIR/'n_channels.csv')
    print("Saving the data")
    data.drop(['n_channels'], axis=1)

    print('Computing the number of channels associated with ')
    print('a given IP address and app...')
    print('IPアドレス毎/app毎のチャネル数を数えている')
    n_chans = data[['ip', 'app', 'channel']].groupby(by=['ip', 'app'])[
        ['channel']].count().reset_index().rename(columns={'channel': 'ip_app_count'})
    print('Merging the channels data with the main data set...')
    data = data.merge(n_chans, on=['ip', 'app'], how='left')
    del n_chans
    gc.collect()
    data['ip_app_count'].astype('uint16').to_csv(WORKDIR/'ip_app_count.csv')
    print("Saving the data")
    data.drop(['ip_app_count'], axis=1)

    print('Computing the number of channels associated with ')
    print('a given IP address, app, and os...')
    print('IPアドレス毎/app毎/os毎のチャネル数を数えている')
    n_chans = data[['ip', 'app', 'os', 'channel']].groupby(by=['ip', 'app', 'os'])[
        ['channel']].count().reset_index().rename(columns={'channel': 'ip_app_os_count'})
    print('Merging the channels data with the main data set...')
    data = data.merge(n_chans, on=['ip', 'app', 'os'], how='left')
    del n_chans
    gc.collect()
    data['ip_app_os_count'].astype('uint16').to_csv(
        WORKDIR/'ip_app_os_count.csv')
    print("Saving the data")
    data.drop(['ip_app_os_count'], axis=1)

    del data
    gc.collect()


def create_LDA_features(data, column_pair):
    col1, col2 = column_pair
    print('pair of %s & %s' % (col1, col2))
    tmp_dict = {}
    for v_col1, v_col2 in zip(data[col1], data[col2]):
        tmp_dict.setdefault(v_col1, []).append(str(v_col2))

    col1_list = list(tmp_dict.keys())
    col2s_of_col1s_list = [[' '.join(tmp_dict[tokun])] for tokun in col1_list]

    dictionary = corpora.Dictionary(col2s_of_col1s_list)
    corpus = [dictionary.doc2bow(tokens) for tokens in col2s_of_col1s_list]
    print('Start learning LDA model')

    model = models.LdaModel(corpus,
                            num_topics=5,
                            id2word=dictionary,
                            random_state=3655
                            )

    print('Saving the model')
    features = np.array(model.get_document_topics(
        corpus, minimum_probability=0))[:, :, 1]

    column_name_list = ["lda_%s_%s_" % (col1, col2) + str(i) for i in range(5)]

    df_features = pd.DataFrame(features, columns=column_name_list)
    df_features[col1] = col1_list

    print("---merging data---")
    print(df_features.head())

    data = pd.merge(data, df_features, on=col1, how='left')
    del df_features
    gc.collect()

    datapath = "lda_" + col1 + "_" + col2 + ".csv"
    data[column_name_list].to_csv(WORKDIR/datapath)

    print("shape of merged data is %s %s " % data[column_name_list].shape)


def get_column_pairs(columns):
    return [(col1, col2) for col1, col2 in itertools.product(columns, repeat=2) if col1 != col2]


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


def lgb_modelfit_nocv(params, dtrain, dvalid, predictors, target='target', objective='binary', metrics='auc',
                      feval=None, early_stopping_rounds=20, num_boost_round=3000, verbose_eval=10, categorical_features=None):
    lgb_params = {
        'boosting_type': 'gbdt',
        'objective': objective,
        'metric': metrics,
        'learning_rate': 0.01,
        # 'is_unbalance': 'true',  #because training data is unbalance (replaced with scale_pos_weight)
        'num_leaves': 31,  # we should let it be smaller than 2^(max_depth)
        'max_depth': -1,  # -1 means no limit
        # Minimum number of data need in a child(min_data_in_leaf)
        'min_child_samples': 20,
        'max_bin': 255,  # Number of bucketed bin for feature values
        'subsample': 0.6,  # Subsample ratio of the training instance.
        'subsample_freq': 0,  # frequence of subsample, <=0 means no enable
        # Subsample ratio of columns when constructing each tree.
        'colsample_bytree': 0.3,
        # Minimum sum of instance weight(hessian) needed in a child(leaf)
        'min_child_weight': 5,
        'subsample_for_bin': 200000,  # Number of samples for constructing bin
        'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
        'reg_alpha': 0,  # L1 regularization term on weights
        'reg_lambda': 0,  # L2 regularization term on weights
        'nthread': 4,
        'verbose': 0,
        'metric': metrics
    }

    lgb_params.update(params)

    print("preparing validation datasets")

    xgtrain = lgb.Dataset(dtrain[predictors].values, label=dtrain[target].values,
                          feature_name=predictors,
                          categorical_feature=categorical_features
                          )
    xgvalid = lgb.Dataset(dvalid[predictors].values, label=dvalid[target].values,
                          feature_name=predictors,
                          categorical_feature=categorical_features
                          )

    evals_results = {}

    bst1 = lgb.train(lgb_params,
                     xgtrain,
                     valid_sets=[xgtrain, xgvalid],
                     valid_names=['train', 'valid'],
                     evals_result=evals_results,
                     num_boost_round=num_boost_round,
                     early_stopping_rounds=early_stopping_rounds,
                     verbose_eval=10,
                     feval=feval)

    n_estimators = bst1.best_iteration
    print("\nModel Report")
    print("n_estimators : ", n_estimators)
    print(metrics+":", evals_results['valid'][metrics][n_estimators-1])

    return bst1


if __name__ == "__main__":
    start = time.time()
    train, test = load_datasets()

    loading_time = time.time() - start
    print("loading time is %s" % loading_time)

    len_train = len(train)
    data = bind_tr_test(train, test)
    data = create_time_features(data)

    create_count_channels_features(data)

    columns = ['app', 'os', 'channel']
    # column_pairs = get_column_pairs(columns)

    for col in columns:
        pair = ('ip', col)
        create_LDA_features(data, pair)

    features = ["ip_app_count", "ip_app_os_count", "lda_ip_app", "lda_ip_channel",
                "lda_ip_os", "n_channels"]

    for feature in features:
        print("merging %s" % feature)
        featurepath = feature + '.csv'
        df_feature = pd.read_csv(WORKDIR/featurepath)
        data = pd.concat([data, df_feature], axis=1)
        del df_feature
        gc.collect()
        print("shape of data is %s %s" % (data.shape))

    train, test = train_test_split(data, len_train)
    sampled_train = negative_down_sampling(
        train, target_variable='is_attributed', random_state=3655)

    del train
    gc.collect()

    print(sampled_train.head())
    print(sampled_train.shape)
    print("="*80)
    print(test.head())
    print(test.shape)

    val = sampled_train[(len_train-25000):len_train]
    train = sampled_train[:(len_train-25000)]

    print("train size: ", len(train))
    print("valid size: ", len(val))
    print("test size : ", len(test))

    target = 'is_attributed'
    predictors = ['app', 'device', 'os', 'channel', 'hour', 'day',
                  'ip_app_count', 'ip_app_os_count',
                  'lda_ip_app_0', 'lda_ip_app_1', 'lda_ip_app_2', 'lda_ip_app_3', 'lda_ip_app_4',
                  'lda_ip_os_0', 'lda_ip_os_1', 'lda_ip_os_2', 'lda_ip_os_3', 'lda_ip_os_4',
                  'lda_ip_channel_0', 'lda_ip_channel_1', 'lda_ip_channel_2', 'lda_ip_channel_3', 'lda_ip_channel_4']

    categorical = ['app', 'device', 'os', 'channel', 'hour', 'day']

    sub = pd.DataFrame()
    test_id = pd.read_csv(INPUTDIR/'test.csv')
    sub['click_id'] = test_id['click_id'].astype('int')
    del test_id
    gc.collect()

    print(sub.head())

    print("Training...")
    start_time = time.time()

    params = {
        'learning_rate': 0.15,
        # 'is_unbalance': 'true', # replaced with scale_pos_weight argument
        'num_leaves': 7,  # 2^max_depth - 1
        'max_depth': 3,  # -1 means no limit
        # Minimum number of data need in a child(min_data_in_leaf)
        'min_child_samples': 100,
        'max_bin': 100,  # Number of bucketed bin for feature values
        'subsample': 0.7,  # Subsample ratio of the training instance.
        'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
        # Subsample ratio of columns when constructing each tree.
        'colsample_bytree': 0.9,
        # Minimum sum of instance weight(hessian) needed in a child(leaf)
        'min_child_weight': 0,
        'scale_pos_weight': 99  # because training data is extremely unbalanced
    }

    bst = lgb_modelfit_nocv(params,
                            train,
                            val,
                            predictors,
                            target,
                            objective='binary',
                            metrics='auc',
                            early_stopping_rounds=30,
                            verbose_eval=True,
                            num_boost_round=30,
                            categorical_features=categorical)

    print('[{}]: model training time'.format(time.time() - start_time))
    del train_df
    del val_df
    gc.collect()

    print("Predicting...")
    sub['is_attributed'] = bst.predict(test[predictors])
    print("writing...")
    sub.to_csv('sub_lgb.csv', index=False)
    print("done...")
