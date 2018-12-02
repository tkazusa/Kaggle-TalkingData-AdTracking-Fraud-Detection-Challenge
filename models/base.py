# -*- encoding: UTF-8 -*-
from abc import abstractclassmethod
from typing import List


class Model:
    @abstractclassmethod
    def train_and_predict(self, train, valid, categorical_features: List[str], target: str, params: dict):
        raise NotImplementedError
