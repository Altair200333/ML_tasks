import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from data_processing import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor


class DataTransformer:
    def __init__(self, use_scaler = False, scaler = None):

        
        if use_scaler or scaler is not None:
            if scaler is not None:
                self.scaler = scaler
            else:
                self.scaler = StandardScaler()
            
            self.use_scaler = True
        else:
            self.scaler = None
            self.use_scaler = False

        self.enc = OneHotEncoder(handle_unknown = 'ignore', sparse=False)

        self.imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

    def fit_encoder(self, X):
        o_cols = X.select_dtypes([object]).columns
        self.enc.fit(X[o_cols])

    def encode(self, X):
        #select object cols
        o_cols = X.select_dtypes([object]).columns

        #encode those and create df
        encoded = self.enc.transform(X[o_cols]) #one_hot_encode(X)
        df = pd.DataFrame(encoded)

        #drop object cols and append encoded
        result = X.drop(columns=o_cols)

        print(result.shape, df.shape)

        result = pd.concat([result.reset_index(drop=True), df.reset_index(drop=True)], axis=1)

        return result

    def imputer_fit(self, X):
        self.imputer.fit(X)

    def imputer_transform(self, X):
        data = self.imputer.transform(X)
        return data

    def scaler_fit(self, X):
        if self.use_scaler:
            self.scaler.fit(X)

    def scaler_transform(self, X):
        if self.use_scaler:
            data = self.scaler.transform(X)

            return data
        else:
            return X