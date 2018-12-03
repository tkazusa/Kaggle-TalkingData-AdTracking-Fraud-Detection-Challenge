# -*- encoding: UTF-8 -*-
import pandas as pd
import numpy as np

from sklearn.feature_extraction import FeatureHasher

"""
http://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-munging/target-encoding.html
"""

class FrequencyEncoder:
    """Encode categorical integer features to numerical features correspondig to its frequency."""
    
    def __init__(self):
        self.categories_ = None
        
    
    def _frequency_encode(self, values):
        """Fit OneHotEncoder to x.
        Parameters
        ----------
        x : a pandas series
        Returns
        -------
        self
        """
        freq_enc = values.groupby(values).size()
        freq_enc = freq_enc / values.shape[0]
        self.freq_enc = freq_enc
        
        
    def fit_transform(self, x, y=None):
        """Fit FrequencyEncoder to X, then transform X.
        Equivalent to fit(X).transform(X) but more convenient.
        Parameters
        ----------
        X : a pandas series
        Returns
        -------
        x_out : a pandas series transformed input.
        """
        return self.fit(x).transform(y)
    
    
    def fit(self, x):
        """Fit FrequencyEncoder to X.
        Parameters
        ----------
        X : a pandas series
        Returns
        -------
        self
        """
        self._frequency_encode(x)
        return self
        
        
    def transform(self, x):
        """Transform X using one-hot encoding.
        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data to encode.
        Returns
        -------
        X_out : a pandas series transformed input.
        """
        x_out = x.map(self.freq_enc).to_frame(name="FE_" + x.name)
        x_out = x_out.fillna(0)
        return x_out
    
    
class HashEncoder:
    """Encode categorical integer features to integer features with hash trick."""
    from sklearn.feature_extraction import FeatureHasher
    
    def __init__(self, n_features=100):
        self.n_features = n_features
        self.FeatureHasher = FeatureHasher(n_features, input_type = 'string')
        
        
    def fit_transform(self, x, y=None):
        """Fit HashEncoder to X, then transform X.
        Equivalent to fit(X).transform(X) but more convenient.
        Parameters
        ----------
        X : a pandas series
        Returns
        -------
        X_out : a pandas DataFrame transformed input.
        """
        return self.fit(x).transform(y)
    
    
    def fit(self, x):
        """Fit HashEncoder to X.
        Parameters
        ----------
        X : a pandas series
        Returns
        -------
        self
        """
        self.FeatureHasher.fit(x.astype('str'))
        return self
        
        
    def transform(self, x):
        """Transform X using one-hot encoding.
        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data to encode.
        Returns
        -------
        X_out : a pandas DataFrame transformed input.
        """
        x_out = self.FeatureHasher.transform(x.astype('str'))
        
        col_names = []
        for i in range(self.n_features):
            col_names.append(x.name + "_" + str(i))
            
        X_out = pd.DataFrame(x_out.A, columns=col_names).astype('int')
        
        return X_out
    
    
class MeanEncoder:
    """Encode categorical integer features to numerical features correspondig to target mean."""
    
    def __init__(self):
        self._mean_enc = None
        self._global_mean = None
    
    
    def _mean_encode(self, feature, target):
        """Fit MeanEncoder to feature corresponding to target.
        Parameters
        ----------
        x : a pandas series
        Returns
        -------
        self
        """
        df = pd.concat([feature, target], axis=1)
        mean_enc = df.groupby(df[feature.name])[target.name].mean()
        self._mean_enc = mean_enc
        
    
    def fit_transform(self, tr_feature, target, tst_feature):
        """Fit MeanEncoder to train feature and target then transform test target.
        Equivalent to fit(X).transform(X) but more convenient.
        Parameters
        ----------
        X : a pandas series
        Returns
        -------
        X_out : a pandas DataFrame transformed input.
        """
        return self.fit(feature, tr_target).transform(tst_feature)
    
    
    def fit(self, feature, target):
        """Fit MeanEncoder to feature and target.
        Parameters
        ----------
        feature : a pd.Series
        target : a pd.Series
        Returns
        -------
        self
        """
        self._grobal_mean = target.mean()
        self._mean_encode(feature, target)
        return self
        
        
    def transform(self, feature):
        """Transform X using mean encoding.
        Parameters
        ----------
        feature : a pd.Series
        Returns
        -------
        x_out : a pandas DataFrame transformed input.
        """
        x_out = feature.map(self._mean_enc).to_frame(name="ME_" + feature.name)
        x_out = x_out.fillna(self._grobal_mean)
        return x_out
        
        
class SmoothingMeanEncoder:
    """Encode categorical integer features to numerical features correspondig to target mean with smoothing."""
    
    def __init__(self):
        self._mean_enc = None
        self._global_mean = None
    
    
    def _mean_encode(self, feature, target):
        """Fit MeanEncoder to feature corresponding to target.
        Parameters
        ----------
        feature: a pandas series
        target: a pandas series
        Returns
        -------
        self
        """
        self._df = pd.concat([feature, target], axis=1)
        self._mean_enc = self._df.groupby(self._df[feature.name])[target.name].mean()
        
        
    def _smoothing_mean_encode(self, feature, target, alpha=100):
        """Fit SmoothingMeanEncoder to feature corresponding to target.
        Parameters
        ----------
        feature: a pandas series
        target: a pandas series
        Returns
        -------
        self
        """
        self._mean_encode(feature, target)
        self._n_objects = self._df.groupby(self._df[feature.name])[target.name].transform('count')
        self._smoothing_mean_enc = (self._mean_enc * self._n_objects + self._grobal_mean * alpha) / (self._n_objects + alpha)

        
    
    def fit_transform(self, tr_feature, target, tst_feature):
        """Fit MeanEncoder to train feature and target then transform test target.
        Equivalent to fit(X).transform(X) but more convenient.
        Parameters
        ----------
        tr_feature : a pandas series
        target: a pandas series
        tst_featire: a pandas series
        Returns
        -------
        x_out : a pandas series transformed input.
        """
        return self.fit(feature, tr_target).transform(tst_feature)
    
    
    def fit(self, feature, target):
        """Fit SmoothingMeanEncoder to feature and target.
        Parameters
        ----------
        feature : a pd.Series
        target : a pd.Series
        Returns
        -------
        self
        """
        self._grobal_mean = target.mean()
        self._smoothing_mean_encode(feature, target)
        return self
        
        
    def transform(self, feature):
        """Transform X using smoothing mean encoding.
        Parameters
        ----------
        feature : a pd.Series
        Returns
        -------
        x_out : a pandas series transformed input.
        """
        x_out = feature.map(self._smoothing_mean_enc).to_frame(name="SME_" + feature.name)
        x_out = x_out.fillna(self._grobal_mean)
        return x_out
    
class ExpandingMeanEncoder:
    """Encode categorical integer features to numerical features correspondig to expanding target mean."""
    
    def __init__(self):
        self._mean_enc = None
        self._global_mean = None
    
    
    def _expanding_mean_encode(self, feature, target):
        """Fit MeanEncoder to feature corresponding to target.
        Parameters
        ----------
        feature: a pandas series
        target: a pandas series
        Returns
        -------
        self
        """
        self._df = pd.concat([feature, target], axis=1)
        self._target_cumsum = self._df.groupby(self._df[feature.name])[target.name].cumsum() - target
        self._target_cumcount = self._df.groupby(self._df[feature.name]).cumcount()
  
        
    
    def fit_transform(self, feature, target):
        """Fit MeanEncoder to train feature and target then transform test target.
        Equivalent to fit(X).transform(X) but more convenient.
        Parameters
        ----------
        tr_feature : a pandas series
        target: a pandas series
        tst_featire: a pandas series
        Returns
        -------
        x_out : a pandas series transformed input.
        """
        return self.fit(feature, target).transform(feature)
    
    
    def fit(self, feature, target):
        """Fit SmoothingMeanEncoder to feature and target.
        Parameters
        ----------
        feature : a pd.Series
        target : a pd.Series
        Returns
        -------
        self
        """
        self._grobal_mean = target.mean()
        self._expanding_mean_encode(feature, target)
        return self
        
        
    def transform(self, feature):
        """Transform X using smoothing mean encoding.
        Parameters
        ----------
        feature : a pd.Series
        Returns
        -------
        x_out : a pandas series transformed input.
        """
        feature["EME_" + feature.name] = self._target_cumsum / self._target_cumcount        
        x_out = feature["EME_" + feature.name].fillna(self._grobal_mean)
        return x_out



