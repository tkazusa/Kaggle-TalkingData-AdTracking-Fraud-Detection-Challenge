# -*- encoding: UTF-8 -*-
from pathlib import Path

import pandas as pd

from features import Feature
# from utils import Logger, logfile_name

# logger = Logger(logfile_name=logfile_name)

Feature.dir = 'features'
Feature.file_prefix = '0'
datadir = Path(__file__).parents[1] / 'data' / 'input'

train = pd.read_csv(datadir/'train.csv.small')
test = pd.read_csv(datadir/'test.csv.small')

train['click_time'] = pd.to_datetime(train['click_time'])
test['click_time'] = pd.to_datetime(test['click_time'])

class TimeInformation(Feature):
    def create_features(self):
        self.train = train
        print(self.train.head())
        print(self.train.shape)
        self.test = test
         
        self.train['day'] = self.train['click_time'].dt.day.astype('uint8')
        self.train['hour'] = self.train['click_time'].dt.hour.astype('uint8')
        self.train['minute'] = self.train['click_time'].dt.minute.astype('uint8')
        self.train['second'] = self.train['click_time'].dt.second.astype('uint8')
        
        self.test['day'] = self.test['click_time'].dt.day.astype('uint8')
        self.test['hour'] = self.test['click_time'].dt.hour.astype('uint8')
        self.test['minute'] = self.test['click_time'].dt.minute.astype('uint8')
        self.test['second'] = self.test['click_time'].dt.second.astype('uint8')

        self.train = self.train.reset_index(drop=True)
        self.test = self.test.reset_index(drop=True)


        print(self.train.head())
        print(self.train.shape)
