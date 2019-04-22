# -*- encoding: UTF-8 -*-
from typing import List, Tuple

import numpy as np
import lightgbm as lgb
import pandas as pd
import time
from lightgbm import Booster

from models import Model


class LightGBM(Model):
    def train_and_predict(self, train, valid, categorical_features: List[str], target: str, params: dict) \
            -> Tuple[Booster, dict]:
        if type(train) != pd.DataFrame or type(valid) != pd.DataFrame:
            raise ValueError('Parameter train and valid must be pandas.DataFrame')

        if list(train.columns) != list(valid.columns):
            raise ValueError('Train and valid must have a same column list')

        classdist = pd.value_counts(train['target'])
        weights = {i: round(np.sum(classdist) / classdist[i]) for i in classdist.index}
        weight_list_train = [weights[i] for i in train['target'].values]
        weight_list_valid = [weights[i] for i in valid['target'].values]

        d_train = lgb.Dataset(train.drop(target, axis=1), label=train[target].values, weight=weight_list_train)
        d_valid = lgb.Dataset(valid.drop(target, axis=1), label=valid[target].values, weight=weight_list_valid)

        eval_results = {}
        model = lgb.train(params['model_params'],
                          d_train,
                          categorical_feature=categorical_features,
                          valid_sets=[d_train, d_valid],
                          valid_names=['train', 'valid'],
                          evals_result=eval_results,
                          **params['train_params'])
        return model, eval_results

    
    def train_without_validation(self, train, weight, categorical_features: List[str], target: str, params: dict, best_iteration: int):
        predictors = train.columns.drop(target)
        if weight is None:
            d_train = lgb.Dataset(train[predictors], label=train[target].values)
        else:
            d_train = lgb.Dataset(train[predictors], label=train[target].values, weight=weight)
        train_params = copy.deepcopy(params['train_params'])
        train_params['num_boost_round'] = best_iteration
        if 'early_stopping_rounds' in train_params:
            del train_params['early_stopping_rounds']
        model = lgb.train(params['model_params'],
                          d_train,
                          categorical_feature = categorical_features,
                          **train_params)
        return model
