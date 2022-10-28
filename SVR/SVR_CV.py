import numpy as np
import os
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
import pandas as pd


def load_data(data_dir):
    values = pd.read_csv(os.path.join(data_dir, 'Value1.csv')
                         , parse_dates=['timestamp'])
    values = values.drop_duplicates(subset="timestamp", keep='first')
    values.index = values['timestamp']
    values = values.reindex(pd.date_range(min(values['timestamp']),
                                          max(values['timestamp']),
                                          freq='S'))
    values = values.drop('timestamp', axis=1)
    values = values.interpolate()
    return values


vibration = load_data('../data')[['vibration']]

train_start_dt = '2021-06-21 07:00:00'
test_start_dt = '2021-06-21 12:00:00'
test_end_dt = '2021-06-21 17:00:00'

train = vibration.copy()[(vibration.index >= train_start_dt) & (vibration.index < test_start_dt)][['vibration']]
test = vibration.copy()[(vibration.index >= test_start_dt) & (vibration.index < test_end_dt)][['vibration']]

scaler = MinMaxScaler()

train['vibration'] = scaler.fit_transform(train)
test['vibration'] = scaler.transform(test)

train_data = train.values
test_data = test.values

timesteps=5

train_data_timesteps = np.array([[j for j in train_data[i:i+timesteps]] for i in range(0, len(train_data)-timesteps+1)])[ :, :, 0]

test_data_timesteps = np.array([[j for j in test_data[i:i+timesteps]] for i in range(0, len(test_data)-timesteps+1)])[:, :, 0]

X_train, y_train = train_data_timesteps[:, :timesteps-1], train_data_timesteps[:, [timesteps-1]]
X_test, y_test = test_data_timesteps[:, :timesteps-1], test_data_timesteps[:, [timesteps-1]]


params = {
    'kernel': ['rbf', 'linear', 'sigmoid'],
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1, 10, 100],
    'epsilon': [0.001, 0.01, 0.1,  1, 10, 100]
    }

grid_search = GridSearchCV(SVR(), params, cv=5, n_jobs=-1,verbose=1)

grid_search.fit(X_train, y_train[:,0])

print("train score - " + str(grid_search.score(X_train, y_train)))
print("test score - " + str(grid_search.score(X_test, y_test)))

print("SVR GridSearch score: "+str(grid_search.best_score_))
print("SVR GridSearch params: ")
print(grid_search.best_params_)