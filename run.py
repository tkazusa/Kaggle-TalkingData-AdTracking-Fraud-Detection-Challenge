# -*- encoding: UTF-8 -*-
import argparse
import itertools
import json
import os
import time
from typing import List

import numpy as np
import pandas as pd
import lightgbm as lgb

from models.lightgbm import LightGBM


def dump_json_log(options, train_results):
    config = json.load(open(options.config))
    results = {
        'training': {
            'trials': train_results,
            'average_train_auc': np.mean([result['train_auc'] for result in train_results]),
            'average_valid_auc': np.mean([result['valid_auc'] for result in train_results]),
            'train_auc_std': np.std([result['train_auc'] for result in train_results]),
            'valid_auc_std': np.std([result['valid_auc'] for result in train_results]),
            'average_time': np.mean([result['time'] for result in train_results])
        },
        'config': config,
    }
    log_path = os.path.join(os.path.dirname(__file__), output_directory,
                            os.path.basename(options.config) + '.result_downsampling.json')
    json.dump(results, open(log_path, 'w'), indent=2)


def load_datasets(feats: List[str]):
    dfs = [pd.read_csv('features/{}_train.csv'.format(f)) for f in feats]
    train = pd.concat(dfs, axis=1)
    train['target'] = train['is_attributed']
    train.drop(['is_attributed', 'click_time'], axis=1, inplace=True)
    
    val = train[train['day'] == 9]
    train = train[train['day'] == 8]

    dfs = [pd.read_csv('features/{}_test.csv'.format(f)) for f in feats]
    test = pd.concat(dfs, axis=1)
    test.drop(['click_time'], axis=1, inplace=True)
    
    return train, val, test


def negative_down_sampling(data, random_state, target_variable):
    positive_data = data[data[target_variable] == 1]
    positive_ratio = float(len(positive_data)) / len(data)
    negative_data = data[data[target_variable] == 0].sample(
        frac=positive_ratio / (1 - positive_ratio), random_state=random_state)
    return pd.concat([positive_data, negative_data])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./config/0.lightgbm.json')
    options = parser.parse_args()
    config = json.load(open(options.config))


    output_directory = 'data/output'
    models = {'lightgbm': LightGBM}
    feats = config['features']

    # Data load
    train, val, test = load_datasets(feats)

    categorical_features = ['ip', 'app', 'device', 'os', 'channel', 'day', 'hour', 'minute', 'second']
    train_results = []

    model = models[config['model']['name']]()

    start_time = time.time()
    sampled_train = negative_down_sampling(train, target_variable='target', random_state=3)

    booster, result = model.train_and_predict(train=sampled_train,
                                              valid=val,
                                              categorical_features=categorical_features,
                                              target='target',
                                              params=config['model'])

    train_results.append({
        'train_auc': result['train']['auc'][booster.best_iteration],
        'valid_auc': result['valid']['auc'][booster.best_iteration],
        'best_iteration': booster.best_iteration,
        'time': time.time() - start_time})

    dump_json_log(options, train_results)
    target = 'target'
    y_pred_prob = booster.predict(test, num_iteration=booster.best_iteration)

