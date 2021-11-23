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


def get_missing_values_info(df):
    total=df.isnull().sum().sort_values(ascending=False)
    percent=total/len(df)*100
    temp=pd.concat([total, percent], axis=1, keys=['Total', '%'])
    return temp.loc[(temp['%']>5)]


def fill_nan(df):
    numeric_data = df.select_dtypes([np.number])

    numeric_data_mean = numeric_data.mean()

    X_filled = df.fillna(numeric_data_mean)
    
    return X_filled, numeric_data_mean

def one_hot_encode(df):
    categorical = df.select_dtypes(include='object')

    processed = pd.get_dummies(categorical)

    encoded = df.drop(categorical.columns.values, axis = 1)
    encoded = pd.concat([encoded, processed], axis=1)

    return encoded