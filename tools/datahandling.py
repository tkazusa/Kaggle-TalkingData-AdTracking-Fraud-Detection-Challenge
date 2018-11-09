# -*- encoding: UTF-8 -*-
#
# Author: taketoshi.kazusa
#
import os
import datetime

import pandas as pd
from sklearn.externals import joblib
from sklearn.model_selection import StratifiedKFold

from tools.util import Util

logfile_name = 'logs/' + str(datetime.date.today().isoformat())+ '.log'



class DataHandling:

    def __init__(self):
        pass

    @classmethod
    def DowsizeBydtypes(self, df):
        df_converted = pd.DataFrame()
        logger = Util.Logger(logfile_name=logfile_name)
        for name in df.columns:
            logger.info('Converting %s' % name)

            if df[name].dtypes == 'int64':
                logger.info('%s is converted from int64 to int32' % name)
                df_converted[name] = df[name].astype('int32')

            elif df[name].dtypes == 'float64':
                df_converted[name] = df[name].astype('float32')
                logger.info('%s is converted from float64 to float32' % name)

            else:
                df_converted[name] = df[name]
                logger.info('%s has nothing to be done' % name)

        return df_converted


    @classmethod
    def KFoldToPickle(self, X, y, num_of_folds, DIR_TO_SAVED):
        """
        :param X: array
        :param y: array
        :param num_of_folds:[int, int, ...]
        :X_train, X_val, y_train, y_val are saved on DIr 'data/n_folds_xx'
        """
        logger = Util.Logger(logfile_name=logfile_name)
        for num_of_fold in num_of_folds:
            logger.info('splitting to %s division data' % num_of_fold)
            skf = StratifiedKFold(n_splits=num_of_fold)
            CV_DIR = os.path.join(DIR_TO_SAVED, 'n_folds_%s/' % num_of_fold)
            for i, (train_idx, test_idx) in enumerate(skf.split(X, y)):
                logger.info('writing %s th cv data')

                X_train, X_val = pd.DataFrame(X[train_idx]), pd.DataFrame(X[test_idx])
                logger.info('X_train data shape %s %s' % X_train.shape)
                logger.info('X_val data shape %s %s' % X_val.shape)

                y_train, y_val = pd.DataFrame(y[train_idx]), pd.DataFrame(y[test_idx])
                logger.info('y_train data shape %s %s' % y_train.shape)
                logger.info('y_val data shape %s %s' % y_val.shape)

                logger.info('saving on %s' % CV_DIR)
                Util.ToPickle(X_train, os.path.join(CV_DIR, 'X_train_%s.pickle' % i))
                self.ToPickle(X_val, os.path.join(CV_DIR, 'X_val_%s.pickle' % i))
                self.ToPickle(y_train, os.path.join(CV_DIR, 'y_train_%s.pickle' % i))
                self.ToPickle(y_val, os.path.join(CV_DIR, 'y_val_%s.pickle' % i))

    @classmethod
    def ToCSVEachColumn(self, df):
        logger = Util.Logger(logfile_name=logfile_name)

        for column in df.columns:
            df[column].to_csv('input/tr_test_' + str(column) + '.csv')
            logger.info('pd.Series of %s is saved' % column)

        logger.info('end save data')

    @classmethod
    def SubsetToFeather(self, df, subsample='head', n_sample=100, filepath=None):
        logger = Util.Logger(logfile_name=logfile_name)
        logger.info('Saving %s subsample data' % subsample)
        if subsample == 'head':
            df = df.head(n_sample)

        elif subsample == 'tail':
            df = df.head(n_sample)
            
        elif subsample == 'random':
            df = df.sample(n_sample)
            df.to_csv(filepath)

        df.to_csv(filepath)
        logger.info('end save data')

