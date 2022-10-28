import math
import numpy as np
import pandas as pd
import os


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