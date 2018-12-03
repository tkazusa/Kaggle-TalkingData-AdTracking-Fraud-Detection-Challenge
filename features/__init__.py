# -*- encoding: UTF-8 -*-
# -*- encoding: UTF-8 -*-
import argparse
import inspect
import hashlib
import time
from abc import ABC, abstractmethod
from pathlib import Path
from contextlib import contextmanager

import pandas as pd


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', '-f', action='store_true', help='Overwrite existing files')
    return parser.parse_args()


def get_features(namespace):
    for k, v in namespace.items():
        print(v)
        print(inspect.isclass(v))
        if inspect.isclass(v) and issubclass(v, Feature) and not inspect.isabstract(v):
            yield v()


def generate_features(namespace, overwrite):
    for f in get_features(namespace):
        if f.train_path.exists() and f.test_path.exists() and not overwrite:
            print(f.name, 'was skipped')
        else:
            f.run().save()


@contextmanager
def timer(name):
    t0 = time.time()
    print('[{}] start'.format(name))
    yield
    print('[{}] done in {} s'.format(name, time.time() - t0))


class Feature(ABC):
    column_prefix = ''
    column_suffix = ''
    file_prefix = ''
    dir = '.'


    def __init__(self):
        self.name = self.__class__.__name__
        self.train = pd.DataFrame()
        self.test = pd.DataFrame()

        file_prefix = self.file_prefix + '.' if self.file_prefix else ''
        self.train_path = Path(self.dir) / '{}{}_train.csv'.format(file_prefix, self.name)
        self.test_path = Path(self.dir) / '{}{}_test.csv'.format(file_prefix, self.name)

    @abstractmethod
    def create_features(self, train_path, valid_path, test_path) -> (str, str, str):
        raise NotImplementedError

    def run(self):
        with timer(self.name):
            self.create_features()
            prefix = self.column_prefix + '_' if self.column_prefix else ''
            suffix = '_' + self.column_suffix if self.column_suffix else ''
            self.train.columns = prefix + self.train.columns + suffix
            self.test.columns = prefix + self.test.columns + suffix
        return self
    

    @abstractmethod
    def create_features(self):
        raise NotImplementedError
    
    def save(self):
        self.train.to_csv(str(self.train_path), index=False)
        self.test.to_csv(str(self.test_path), index=False)
