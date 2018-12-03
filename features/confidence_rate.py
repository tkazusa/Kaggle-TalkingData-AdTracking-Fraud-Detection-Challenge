# -*- encoding: UTF-8 -*-
from pathlib import Path

import numpy as np
import pandas as pd

from features import Feature


Feature.dir = 'features'
Feature.file_prefix = '1'
datadir = Path(__file__).parents[1] / 'data' / 'input'

train = pd.read_csv(datadir/'train.csv.small')
test = pd.read_csv(datadir/'test.csv.small')

test['click_time'] = pd.to_datetime(test['click_time'])
train['click_time'] = pd.to_datetime(train['click_time'])

ATTRIBUTION_CATEGORIES = [        
    # V1 Features #
    ###############
    ['ip'], ['app'], ['device'], ['os'], ['channel'],
    
    # V2 Features #
    ###############
    ['app', 'channel'],
    ['app', 'os'],
    ['app', 'device'],
    
    # V3 Features #
    ###############
    ['channel', 'os'],
    ['channel', 'device'],
    ['os', 'device']
]


class ConfidenceRate(Feature):
    def create_features(self):
        print(train.head())
        print(train.shape)

        # Find frequency of is_attributed for each unique value in column
        for cols in ATTRIBUTION_CATEGORIES:
            
            # New feature name
            new_feature = '_'.join(cols)+'_confRate'    
            
            # Perform the groupby
            group_object = train.groupby(cols)
            
            # Group sizes    
            group_sizes = group_object.size()
            log_group = np.log(100000) # 1000 views -> 60% confidence, 100 views -> 40% confidence 
            print(">> Calculating confidence-weighted rate for: {}.\n   Saving to: {}. Group Max /Mean / Median / Min: {} / {} / {} / {}".format(
                cols, new_feature, 
                group_sizes.max(), 
                np.round(group_sizes.mean(), 2),
                np.round(group_sizes.median(), 2),
                group_sizes.min()
            ))
            
            # Aggregation function
            def rate_calculation(x):
                """Calculate the attributed rate. Scale by confidence"""
                rate = x.sum() / float(x.count())
                conf = np.min([1, np.log(x.count()) / log_group])
                return rate * conf
            
            # Perform the merge
            self.train[new_feature] = train.merge(
                group_object['is_attributed']. \
                apply(rate_calculation). \
                reset_index(). \
                rename(
                    index=str,
                    columns={'is_attributed': new_feature}
                )[cols + [new_feature]],
                on=cols, how='left'
            )[new_feature]
    
            self.test[new_feature] = test.merge(
                group_object['is_attributed']. \
                apply(rate_calculation). \
                reset_index(). \
                rename(
                    index=str,
                    columns={'is_attributed': new_feature}
                )[cols + [new_feature]],
                on=cols, how='left'
            )[new_feature]

        print(self.train.head())
        print(self.train.shape)
