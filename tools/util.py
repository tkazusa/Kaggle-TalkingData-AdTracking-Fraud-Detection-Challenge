# -*- encoding: UTF-8 -*-
# -*- coding: utf-8 -*-
#
# Author: taketoshi.kazusa
#
import os
import datetime
import json
import time
from contextlib import contextmanager

import requests
import pandas as pd
from sklearn.externals import joblib
from sklearn.model_selection import StratifiedKFold

logfile_name = "logs/" + str(datetime.date.today().isoformat())+ ".log"


class Util:
    
    def __init__(self):
        pass

    @classmethod
    def mkdir(self, dr):
        if not os.path.exists(dr):
            os.makedirs(dr)

    @classmethod
    def mkdir_file(self, path):
        dr = os.path.dirname(path)
        if not os.path.exists(dr):
            os.makedirs(dr)

    @classmethod
    def dump(self, obj, filename, compress=0):
        self.mkdir_file(filename)
        joblib.dump(obj, filename, compress=compress)

    @classmethod
    def dumpc(self, obj, filename):
        self.mkdir_file(filename)
        self.dump(obj, filename, compress=3)

    @classmethod
    def load(self, filename):
        return joblib.load(filename)

    @classmethod
    def read_csv(self, filename, sep=",", header = None, compression=None, chunksize=None):
        return pd.read_csv(filename, header = header, compression=compression, chunksize=chunksize, sep=sep)

    @classmethod
    def ToPickle(self, df, filename, index=False, sep=","):
        self.mkdir_file(filename)
        df.to_pickle(filename, sep=sep, index=index)

    @classmethod
    def nowstr(self):
        return str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M"))

    @classmethod
    def nowstrhms(self):
        return str(datetime.datetime.now().strftime("%H-%M-%S"))

    @classmethod
    def Logger(self, logfile_name):
        import logging
        import daiquiri

        log_fmt = '%(asctime)s %(filename)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s '
        daiquiri.setup(level=logging.DEBUG,
                       outputs=(
                           daiquiri.output.Stream(formatter=daiquiri.formatter.ColorFormatter(fmt=log_fmt)),
                           daiquiri.output.File(logfile_name, level=logging.DEBUG)
                       ))
        return daiquiri.getLogger(__name__)


def send_line_notification(message, config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    line_token = config['line_token']  # 終わったら無効化する
    line_notify_api = 'https://notify-api.line.me/api/notify'
    message = "\n{}".format(message)
    payload = {'message': message}
    headers = {'Authorization': 'Bearer {}'.format(line_token)}  # 発行したトークン
    requests.post(line_notify_api, data=payload, headers=headers)


@contextmanaer
def timer(name):
    start = time.time()
    print(f'[{name}] start')
    yield
    print(f'[{}name]
