import os
import warnings
import matplotlib.pyplot as plt
import numpy as np
from time import time
from prophet import Prophet
import pandas as pd
import random
import datetime as dt
import math

from datetime import datetime

from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from copy import deepcopy
import matplotlib.pyplot as plt
import statsmodels.api as sm
# from common.utils import load_data, mape, rmse, compar

def load_data(data_dir):
    energy = pd.read_csv(os.path.join(data_dir, 'accelerometer.csv'))
    return energy


vibration_x = load_data('/home/sachin/Downloads')[['x']]

from pandas.plotting import autocorrelation_plot
autocorrelation_plot(vibration_x)
vibration_x.hist()
plt.show()


# result = sm.tsa.stattools.adfuller(vibration_x["x"])
# print(f'ADF Statistic: {result[0]}')
# print(f'n_lags: {result[1]}')
# print(f'p-value: {result[1]}')
# for key, value in result[4].items():
#     print('Critial Values:')
#     print(f'   {key}, {value}')
#
# if result[1] < 0.05 and result[0] < result[4]['5%']:
#     print("Stationary")
# else:
#     print("Non Stationary")