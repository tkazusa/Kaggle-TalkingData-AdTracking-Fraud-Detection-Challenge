# TalkingDataAdTrackingFraudDetectionChallenge

## Data resampling
- scripts/convert_to_small.pyでdownsampling
- データサイズがXXXからXXXに

## EDA
- https://www.kaggle.com/yuliagm/talkingdata-eda-plus-time-patterns
https://github.com/flowlight0/talkingdata-adtracking-fraud-detection/tree/6823c09f8dc0fa47a75a1a4e3cc9c24ae03dd7c8



## サイズの大きなデータに対する取り扱い
### データサイズが大きい場合の課題点
- モデリング時
  - メモリに一度に乗らない
  - 特徴量作成や学習に時間がかかる
  - 全期間＆初期特徴量だけでどの程度の量？Pandasでの読み込み時間は？
  -

- 実案件と同様に、原則としてデータを全部一度に使う必要があるのかを考え、サブサンプリングすることを考慮しながら取り組む
- とは言え、一旦は全データを見なければどの部分を使う/使わないの判断は出来ない→ひとまずBQに突っ込んで見るのが良い？
- Kaggle的にはデータによるアンサンブルも考えられる。


- 2nd place solution
  - https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/discussion/56328
  - In the feature engineering phase, new features were extracted from the entire data and merged into the sub-sampled data. In the training phase, only the sub-sampled training samples were used so it was about 10+ times faster than directly training the entire data. 


- データ型の選択
- データのサブサンプリング
- 不均衡データでのダウンサンプリング

### データ型の選択
pandasのDataFrameはDefaultでは倍精度でデータが格納されている。float32やint32、categoricalのデータ型ののとり得る範囲内に収まるようであれば、データ型を変換するだけでもデータサイズを削減できる。
```
例を出す
```

```
@classmethod
def change_dtypes(self, df):
  df_converted = pd.DataFrame()
  logger = Util.Logger(logfile_name=logfile_name)
  for name in df.columns:
    logger.info('Converting %s' % name)
    
      if df[name].dtypes == 'int64':
        logger.info('%s is converted from int64 to int32' % name)
        df_converted[name] = df[name].astype('int32')

      elif df[name].dtypes == 'float64':
        df_converted[name] = df[name].astype('float32')
        logger.info('%s is converted from float64 to float32' % name)

      else:
        df_converted[name] = df[name]
        logger.info('%s has nothing to be done' % name)
```


### EDAによるデータの削減
- 

## 不均衡なデータの取扱
