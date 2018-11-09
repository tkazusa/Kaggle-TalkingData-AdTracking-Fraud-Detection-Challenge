# -*- encoding: UTF-8 -*-
import gc

import pandas as pd
import numpy as np

from tools.util import Util
from tools.encoding import MeanEncoder, HashEncoder, FreqencyEncoder

from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

if __name__ == "__main__":
    X_train = pd.read_pickle('input/org/MeanEncoded/train.pkl')
    y_train = pd.read_pickle('input/org/selected_train.pkl')['is_attributed']

    X_val = pd.read_pickle('input/org/MeanEncoded/val.pkl')
    y_val = pd.read_pickle('input/org/selected_val.pkl')['is_attributed']

    X_test = pd.read_pickle('input/org/MeanEncoded/test.pkl')


    rf = RandomForestClassifier(n_estimators=30,
                                max_depth=10,
                                n_jobs=10,
                                bootstrap=True,
                                max_features=3, 
                                class_weight="balanced")
    rf.fit(X_train, y_train)
    print('model fitted')

    pred_val = rf.predict(X_val)
    fpr, tpr, thresholds = metrics.roc_curve(y_val, pred_val)
    auc = metrics.auc(fpr, tpr)
    print(auc)

    pred_test = rf.predict(X_test)
    my_submission = pd.DataFrame()
    my_submission["click_id"] = pd.read_csv('input/test.csv')["click_id"]
    my_submission["is_attributed"] = pred_test
    my_submission.to_csv("input/ME_RF_local%s.csv" % auc, index=False)
