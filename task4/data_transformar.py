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


def remove_outliers(X):
    ids = X[((X['GrLivArea'] > 4000) & (X["SalePrice"] < 300000)) | (X["LotArea"] > 100000)].index

    return X.drop(ids)


class DataTransformer:
    def __init__(self, scaler=None):
        if scaler is not None:
            self.scaler = scaler
        else:
            self.scaler = StandardScaler()

        self.enc = OneHotEncoder(handle_unknown='ignore', sparse=False)

        self.imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

    def fit_encoder(self, X):
        o_cols = X.select_dtypes([object]).columns
        self.enc.fit(X[o_cols])

    def encode(self, X):
        # select object cols
        o_cols = X.select_dtypes([object]).columns

        # encode those and create df
        encoded = self.enc.transform(X[o_cols])  # one_hot_encode(X)
        df = pd.DataFrame(encoded)

        # drop object cols and append encoded
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
        self.scaler.fit(X)

    def scaler_transform(self, X):
        data = self.scaler.transform(X)

        return data

    def nums_to_cats(self, X):
        num_to_cats = ["HalfBath", "KitchenAbvGr", "Fireplaces", "FullBath", "OverallCond", "OverallQual",
                       "TotRmsAbvGrd", "MSSubClass", "MoSold"]

        for feat in num_to_cats:
            X[feat] = X[feat].apply(str).astype("object")

        return X

    def fillna(self, X):
        X['MSZoning'] = X['MSZoning'].fillna(X['MSZoning'].mode()[0])

        X["LotFrontage"] = X.groupby("Neighborhood")["LotFrontage"].transform(
            lambda x: x.fillna(x.median()))

        # категории у которых остуствие значения означает == 0, так если машин в гараже None, то их наверно 0)
        zero_nan_cols = ['GarageYrBlt', 'GarageArea', 'GarageCars', "MasVnrArea", 'BsmtFinSF1', 'BsmtFinSF2',
                         'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'BedroomAbvGr', "YrSold",
                         "YearRemodAdd"]

        for col in zero_nan_cols:
            X[col] = X[col].fillna(0)

        for col in X.columns:
            if X[col].dtype == "object":
                X[col] = X[col].fillna("None")
                X[col] = X[col].astype("object")
