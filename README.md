# TalkingDataAdTrackingFraudDetectionChallenge
## Setup
Dockerfileの中でgit cloneする。
```
docker compose
```
Data donwload from the kaggle competition page with kaggle api command.
```
kaggle competitions download -c talkingdata-adtracking-fraud-detection -p /home/ec2-user/SageMaker/TalkingDataAdTrackingFraudDetectionChallenge/input
unzip -jl '/home/ec2-user/SageMaker/TalkingDataAdTrackingFraudDetectionChallenge/input/*.zip' 
```

## What you learn from this kernel
- Approach to large size of data
- Classification for inbalamce data

## Approach 
1. pandasデータ型指定によるメモリ使用量の削減 
2. 全データを用いたシンプル特徴量作成
3. LDAを用いたカテゴリカルデータの埋め込み
4. 学習時のNegative down samplingによるサンプルサイズ削減とクラス不均衡へ対応
