# TalkingData AdTracking Fraud Detection Challenge
A brief solution for [TalkingData AdTracking Fraud Detection Challenge]

## Requirements
- docker >= 17.03

## Getting started
### Build docker image 
```
docker build -t <image name> .
docker run -it -p 8888:8888 --name <container name> <image name>
```

### Setup kaggle api credential
Download kaggle.json and place in the location: ~/.kaggle/kaggle.json.

See details: https://github.com/Kaggle/kaggle-api


### Download and unzip datasets from competition page
Data donwload from the kaggle competition page with kaggle api command.
```
mkdir $HOME/input
cd ./input
kaggle competitions download -c talkingdata-adtracking-fraud-detection
unzip '*.zip'
```

### Run jupyter lab
```
jupyter lab --ip 0.0.0.0 --allow-root
```

## What you learn from this kernel
- Approach to large size of data.
  - Memory usage reduction by converting dtypes of pd.DataFrame.
  - Negative downsampling for extremely imbalance data.
- Classification for imbalance data.
  - Negative downsampling for extremely imbalance data.
- Categorical feature embedding
  - Embedding by using LDA.

See a [notebook](https://github.com/tkazusa/TalkingDataAdTrackingFraudDetectionChallenge/blob/master/notebooks/kernel.ipynb) for details.
