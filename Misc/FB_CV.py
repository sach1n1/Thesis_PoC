import itertools
import sys

import pandas as pd
import warnings

import matplotlib.pyplot as plt
import warnings
from contextlib import redirect_stdout
import seaborn as sns
import numpy as np
from pandas import to_datetime, DataFrame
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from sklearn.preprocessing import MinMaxScaler
import datetime as dt
from common.utils import load_data, mape, create_features, rmse
from matplotlib import pyplot
from time import time

warnings.simplefilter('ignore')

energy = load_data('../data')[['vibration']]


train = energy.copy()[(energy.index >= train_start_dt) & (energy.index < test_start_dt)][['vibration']]
test = energy.copy()[(energy.index >= test_start_dt) & (energy.index < test_end_dt)][['vibration']]
train = train.reset_index().rename(columns={'index': 'ds', 'vibration': 'y'})

test = test.reset_index().rename(columns={'index': 'ds', 'vibration': 'y'})
train['ds'] = to_datetime(train['ds'])
test['ds'] = to_datetime(test['ds'])

changepoint_prior_scale = np.arange(0.001, 0.01, 0.001)
seasonality_prior_scale= np.arange(0.1, 1.0, 0.1)

param_grid = {
    'changepoint_prior_scale': np.array(changepoint_prior_scale),
    'seasonality_prior_scale': np.array(seasonality_prior_scale),
}
#
# # Generate all combinations of parameters
all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
rmses = []  # Store the RMSEs for each params here

for params in all_params:
    m = Prophet(**params).fit(train)  # Fit model with given params
    df_cv = cross_validation(m, horizon='2 hours', parallel="processes")
    df_p = performance_metrics(df_cv, rolling_window=1)
    rmses.append(df_p['rmse'].values[0])

tuning_results = pd.DataFrame(all_params)
tuning_results['rmse'] = rmses

best_params = all_params[np.argmin(rmses)]
print(best_params)



#
# start = time()
#
# model = Prophet()
#
# model.fit(train)
#
# future = list()
#
# future = test['ds']
#
# future = DataFrame(future)
#
# forecast = model.predict(future)
#
# print('MAPE for training data: ' + str(round(mape(forecast['yhat'], test['y'])*100, 2)) + '%' + "\n")
# print('RMSE for training data: ' + str(round(rmse(forecast['yhat'], test['y'])*100, 2)) + '%' + "\n")
#
# print(round((time() - start) / 60, 2), 'minutes')