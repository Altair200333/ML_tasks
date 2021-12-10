import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from data_processing import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import scipy.stats as stats


def remove_outliers(X, lot_area=False):
    ids = None
    if not lot_area:
        # ids = X[((X['GrLivArea'] > 4000) & (X["SalePrice"] < 300000)) | (X["LotArea"] > 100000) | (X["LotFrontage"] > 300)].index
        ids = X[((X['GrLivArea'] > 4000) & (X["SalePrice"] < 300000)) | (X["LotArea"] > 100000)].index
    else:
        ids = X[((X['GrLivArea'] > 4000) & (X["SalePrice"] < 300000))].index
    print(ids)
    return X.drop(ids)


def remove_outliers_split(X, y):
    ids = X[((X['GrLivArea'] > 4000) & (np.expm1(y) < 300000)) | (X["LotArea"] > 100000)].index
    print(ids)
    return X.drop(ids), y.drop(ids)


class DataTransformer:
    def __init__(self, scaler=None):
        if scaler is not None:
            self.scaler = scaler
        else:
            self.scaler = StandardScaler()

        self.enc = OneHotEncoder(handle_unknown='ignore', sparse=False)

        self.imputer = IterativeImputer(max_iter=10,
                                        random_state=0)  # SimpleImputer(missing_values=np.nan, strategy='mean')

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
        num_to_cats = ["MoSold", "YrSold"]

        for feat in num_to_cats:
            X[feat] = X[feat].apply(str).astype("object")

        return X

    def fillna(self, X):
        #X = X[(X.MSZoning != 'C (all)') & (X.MSZoning != 'I (all)') & (X.MSZoning != 'A (agr)')]

        X['Exterior1st'] = X['Exterior1st'].fillna(X['Exterior1st'].mode()[0])
        X['Exterior2nd'] = X['Exterior2nd'].fillna(X['Exterior2nd'].mode()[0])
        X['MSZoning'] = X['MSZoning'].fillna(X['MSZoning'].mode()[0])
        # X["LotFrontage"] = X.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))

        dist_cols = ["LotFrontage", "GarageYrBlt"]
        for col in dist_cols:
            missing = X[X[col].isna()][col]
            not_missing = X[X[col].notnull()][col]

            params = stats.johnsonsu.fit(not_missing)

            r = stats.johnsonsu.rvs(params[0], params[1], params[2], params[3], size=missing.shape[0])
            X[col].loc[missing.index] = r

        # категории у которых остуствие значения означает == 0, так если машин в гараже None, то их наверно 0)
        zero_nan_cols = ['GarageArea', 'GarageCars', "MasVnrArea", 'BsmtFinSF1', 'BsmtFinSF2',
                         'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'BedroomAbvGr',
                         "1stFlrSF", "2ndFlrSF"]

        for col in zero_nan_cols:
            X[col] = X[col].fillna(0)

        X['Functional'] = X['Functional'].fillna('Typ')  # Typical Functionality
        X['Electrical'] = X['Electrical'].fillna("SBrkr")  # Standard Circuit Breakers & Romex
        X['KitchenQual'] = X['KitchenQual'].fillna("TA")  # Typical/Average

        for col in X.columns:
            if X[col].dtype == "object":
                X[col] = X[col].fillna("None")
                X[col] = X[col].astype("object")

    def drop_columns(self, X):
        cats_to_drop = ["Utilities", "YearRemodAdd"]
        X = X.drop(cats_to_drop, axis=1)
        return X

    def prepare(self, X):
        X = self.drop_columns(X)
        self.nums_to_cats(X)
        self.fillna(X)
        return X

    def fit(self, X):
        num_candidates = list(X.dtypes[X.dtypes != "object"].index.values)
        self.imputer_fit(X[num_candidates])

        numeric = self.imputer_transform(X[num_candidates])

        # self.scaler_fit(numeric)

        self.fit_encoder(X)

    def transform(self, X, encode=True):
        num_candidates = list(X.dtypes[X.dtypes != "object"].index.values)
        X[num_candidates] = self.imputer_transform(X[num_candidates])

        X['TotalSF'] = X['TotalBsmtSF'] + X['1stFlrSF'] + X['2ndFlrSF']
        X['Total_sqr_footage'] = (X['BsmtFinSF1'] + X['BsmtFinSF2'] + X['1stFlrSF'] + X['2ndFlrSF'])
        X['Total_porch_sf'] = (X['OpenPorchSF'] + X['3SsnPorch'] + X['EnclosedPorch'] + X['ScreenPorch'] + X['WoodDeckSF'])

        # X[num_candidates] = self.scaler_transform(X[num_candidates])

        if encode:
            X = self.encode(X)

        return X
