# TalkingDataAdTrackingFraudDetectionChallenge
## 課題
- 大きなサンプルサイズ
- 極端なクラス不均衡なデータに対する二値分類

## アプローチ
1. pandasデータ型指定によるメモリ使用量の削減 
2. 全データを用いたシンプル特徴量作成
3. LDAを用いたカテゴリカルデータの埋め込み
4. 学習時のNegative down samplingによるサンプルサイズ削減とクラス不均衡へ対応


## pandasデータ型指定によるメモリ使用量の削減
地味に効いてくるのがデータ型の指定。TalkingDataの特徴量は何かしらのidが振られたものが多く、そのまま読み込むとint64で読み込まれる。
読み込み時に型を指定するとメモリ消費が抑えられ読み込みが早くなる。今回のデータではOriginalのtrainデータの読み込み時に型を指定するだけで7分から2分に削減することができた。

```python:reduce_mem_usage.py
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(
        100 * (start_mem - end_mem) / start_mem))

    return df


def import_data(file):
    """create a dataframe and optimize its memory usage"""
    df = pd.read_csv(file, parse_dates=True, keep_date_col=True)
    df = reduce_mem_usage(df)
    return df
```

これによって約7割のメモリ使用量削減。
```
--------------------------------------------------------------------------------
train
Memory usage of dataframe is 11285.64 MB
Memory usage after optimization is: 3721.47 MB
Decreased by 67.0%
--------------------------------------------------------------------------------
test
Memory usage of dataframe is 1003.52 MB
Memory usage after optimization is: 323.35 MB
Decreased by 67.8%
```

## 全データを用いたシンプル特徴量作成
- ベーシックな処理
  - five raw categorical features (ip, os, app, channel, device)  （単純に型をカテゴリ化）
  - time categorical features (day, hour) 
  - some count features 
- web広告配信データ特有の特徴量
  - five raw categorical features (ip, os, app, channel, device) に対し、以下の特徴量を作成 (全組み合わせ2^5 -1 = 31通り)
  - click count within next one/six hours  (直後1 or 6時間以内のクリック数)
  - forward/backward click time delta  (前後クリックまでの時差)
  - average attributed ratio of past click (過去のCVレート)

```python
def create_count_channels_features(data):
    print("Creating new count features: 'n_channels', 'ip_app_count', 'ip_app_os_count'...")

    print('Computing the number of channels associated with ')
    print('a given IP address within each hour...')
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
```

### LDAを用いたカテゴリカルデータの埋め込み
今回のデータはipやosなど、多数のカテゴリをを抱える特徴量がある。それ単体でも特徴なり得るが、任意のカテゴリがどのような意味を持つかについて、他の特徴の各カテゴリとの共起から情報を得る。

```python
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
```



### 不均衡データの取扱
サンプルサイズの削減とクラス不均衡な二値分類への対応としてNegativeDownSamplingを使用した。
```python
def negative_down_sampling(data, random_state, target_variable):
    positive_data = data[data[target_variable] == 1]
    positive_ratio = float(len(positive_data)) / len(data)
    negative_data = data[data[target_variable] == 0].sample(
        frac=positive_ratio / (1 - positive_ratio), random_state=random_state)
    return pd.concat([positive_data, negative_data])
```

