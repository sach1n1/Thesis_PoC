import ast
import math

import numpy as np
import pandas as pd
import os

from collections import UserDict


def create_features(df, label=None):
    """
    Creates time series features from datetime index.
    """
    df = df.copy()
    df['date'] = df.index
    df['second'] = df['date'].dt.second
    df['minute'] = df['date'].dt.minute
    df['hour'] = df['date'].dt.hour
    df['dayofweek'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['dayofyear'] = df['date'].dt.dayofyear
    df['dayofmonth'] = df['date'].dt.day
    df['weekofyear'] = df['date'].dt.weekofyear

    X = df[['second', 'minute', 'hour', 'dayofweek', 'quarter', 'month', 'year',
            'dayofyear', 'dayofmonth', 'weekofyear']]
    if label:
        y = df[label]
        return X, y
    return X


def load_data(data_dir):
    energy = pd.read_csv(os.path.join(data_dir, 'Value1.csv')
                         , parse_dates=['timestamp'])
    energy = energy.drop_duplicates(subset="timestamp", keep='first')
    energy.index = energy['timestamp']
    energy = energy.reindex(pd.date_range(min(energy['timestamp']),
                                          max(energy['timestamp']),
                                          freq='S'))
    energy = energy.drop('timestamp', axis=1)
    energy = energy.interpolate()
    return energy


def mape(predictions, actuals):
    """Mean absolute percentage error"""
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    return (np.absolute(predictions - actuals) / actuals).mean()


def rmse(predictions, actuals):
    """Mean absolute percentage error"""
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    MSE = np.square(np.subtract(actuals, predictions)).mean()
    return math.sqrt(MSE)

def compar(file1,file2,stat):
    dictionary1 = {}
    file_list = [file1,file2]
    for file in file_list:
        f = open(file, 'r')
        data = f.read()
        dictionary1[f"{file}"] = ast.literal_eval(data)
        f.close()
    for key in dictionary1[file1]:
        print(f"{file1}: {dictionary1[file1][key][stat]} {file2}: {dictionary1[file2][key][stat]}")


def create_evaluation_df(predictions, test_inputs, H, scaler):
    """Create a data frame for easy evaluation"""
    eval_df = pd.DataFrame(predictions, columns=['t+' + str(t) for t in range(1, H + 1)])
    eval_df['timestamp'] = test_inputs.dataframe.index
    eval_df = pd.melt(eval_df, id_vars='timestamp', value_name='prediction', var_name='h')
    eval_df['actual'] = np.transpose(test_inputs['target']).ravel()
    eval_df[['prediction', 'actual']] = scaler.inverse_transform(eval_df[['prediction', 'actual']])
    return eval_df