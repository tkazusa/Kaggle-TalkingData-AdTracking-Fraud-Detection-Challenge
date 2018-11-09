# -*- encoding: UTF-8 -*-
from datetime import date
import gc

import pandas as pd
import numpy as np

from tools.util import Util
from tools.datahandling import DataHandling

logfile_name = 'logs/test_test_S_concat_' + str(date.today().isoformat()) + '.log'
logger = Util.Logger(logfile_name=logfile_name)

TRAIN_PATH = 'input/train.pkl'
TEST_PATH = 'input/test.pkl'
TEST_S_PATH = 'input/test_supplement.pkl'


if __name__ == "__main__":
    logger.info("load start")
    
    df_test = pd.read_pickle(TEST_PATH)
    logger.info('test size %s %s' % df_test.shape)

    df_test_s = pd.read_pickle(TEST_S_PATH)
    logger.info('test supplement size %s %s' % df_test_s.shape)

    df_test["data_set"] = int(1)
    df_test_s["data_set"] = int(2)
    df_test_s.drop("click_id", axis=1)

    keys = list(df_test_s.columns)
    df = pd.merge(df_test, df_test_s, how="outer", on=keys)
    logger.info('merged data size %s %s' % df.shape)

    del df_test, df_test_s
    gc.collect()

    df["is_attributed"] = 999

    df_train = pd.read_pickle(TRAIN_PATH)
    df_train["data_set"] = int(0)
    logger.info('train data size %s %s' % df_train.shape)
    df = pd.concat([df_train, df]) 
    logger.info('concatinated data size %s %s' % df.shape)

    df = DataHandling.DowsizeBydtypes(df)
    logger.info('Data size decreased')

    logger.info('saving data')
    df.to_pickle("input/tr_test_concat.pkl")
    for column in df.columns:
        df[column].to_csv("input/tr_test_" + str(column) + ".csv", header=[column], index=False)
        logger.info('pd.Series of %s is saved' % column)
    logger.info('end save data')
