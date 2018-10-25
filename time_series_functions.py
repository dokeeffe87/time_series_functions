"""Basic function for time series feature engineering"""


# import libraries

from __future__ import division

import os
import pandas as pd
import numpy as np
import time
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

# TODO: Add comments.  Add support for different window types


def make_col_name(col, win_, stat):
    return col + '_' + str(win_) + '_' + stat


def moving_averages(df, col, list_of_windows):
    for win_ in list_of_windows:
        win_col = make_col_name(col, win_, 'MA')
        df[win_col] = df[col].rolling(window=win_).mean()
    return df


def exponential_moving_averages(df, col, spans_list):
    for span_ in spans_list:
        span_col = make_col_name(col, span_, 'EMA')
        df[span_col] = pd.Series.ewm(df[col], span=span_).mean()

    return df


def moving_medians(df, col, list_of_windows):
    for win_ in list_of_windows:
        win_col = make_col_name(col, win_, 'MED')
        df[win_col] = df[col].rolling(window=win_).mean()
    return df


def moving_std(df, col, list_of_windows):
    for win_ in list_of_windows:
        win_col = make_col_name(col, win_, 'STD')
        df[win_col] = df[col].rolling(window=win_).std()
    return df


def moving_max(df, col, list_of_windows):
    for win_ in list_of_windows:
        win_col = make_col_name(col, win_, 'MAX')
        df[win_col] = df[col].rolling(window=win_).max()
    return df


def moving_min(df, col, list_of_windows):
    for win_ in list_of_windows:
        win_col = make_col_name(col, win_, 'MIN')
        df[win_col] = df[col].rolling(window=win_).min()
    return df


def moving_sum(df, col, list_of_windows):
    for win_ in list_of_windows:
        win_col = make_col_name(col, win_, 'SUM')
        df[win_col] = df[col].rolling(window=win_).sum()
    return df


def moving_quantiles(df, col, list_of_windows, q_list, interpolation_list):
    for win_ in list_of_windows:
        for interpolation in interpolation_list:
            for q in q_list:
                win_col = make_col_name(col, win_, str(q) + '_QUANTILE_' + interpolation)
                df[win_col] = df[col].rolling(window=win_).apply(lambda x: pd.Series(x).quantile(q=q, interpolation=interpolation))
    return df

