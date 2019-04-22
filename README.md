# TalkingDataAdTrackingFraudDetectionChallenge
## Getting started
### Install required packages 
For SageMaker notebook instance.
```
source activate python3
conda install --channel conda-forge --yes --file requirements.txt
source deactivate 
```


### Setup kaggle api credential
Download kaggle.json and place in the location: ~/.kaggle/kaggle.json.

See details: https://github.com/Kaggle/kaggle-api


### Download and unzip datasets from competition page
Data donwload from the kaggle competition page with kaggle api command.
```
pip install kaggle
cd <REPOSITORY DIRECTORY>
kaggle competitions download -c talkingdata-adtracking-fraud-detection -p ./input
unzip -jl './input/*.zip'
```

## What you learn from this kernel
- Approach to large size of data
- Classification for inbalamce data

## Approach 
1. pandasデータ型指定によるメモリ使用量の削減 
2. 全データを用いたシンプル特徴量作成
3. LDAを用いたカテゴリカルデータの埋め込み
4. 学習時のNegative down samplingによるサンプルサイズ削減とクラス不均衡へ対応
