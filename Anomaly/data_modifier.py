import random

from common.utils import load_data, rmse
import pandas as pd
from time import time
import numpy as np
from copy import deepcopy

y_actual = load_data('/home/sachin/Thesis/Anomaly')[['y_actual']]
y_pred = load_data('/home/sachin/Thesis/Anomaly')[['y_pred']]


test_start_dt = '2021-05-27 18:00:00'
test_end_dt = '2021-05-27 19:00:00'

train = y_actual.copy()[(y_actual.index >= test_start_dt) & (y_actual.index < test_end_dt)][['y_actual']]
train_pred = y_pred.copy()[(y_pred.index >= test_start_dt) & (y_pred.index < test_end_dt)][['y_pred']]
selects = random.sample(range(len(train)), round(len(train)/2))

def modifyList(pred, mod_value):
    for element in selects:
        pred[element] = mod_value * pred[element]
        #pred[ind] = pred[selects]
    return pred


pred = train_pred["y_pred"].to_list()

orig = train["y_actual"].to_list()

mod = train["y_actual"].to_list()

mod_values = []

mod_values = np.arange(1.2, 3.0, 0.1)

mod_list = {}

for mod_value in mod_values:
    mod = deepcopy(orig)
    mod_return = modifyList(mod, mod_value)
    mod_list[mod_value] = deepcopy(mod_return)

line = ""

print(len(mod_list[1.2]))

for element in range(0, len(mod_list[1.2])):
    for mod_value in mod_values:
        line = line + ", " + str(mod_list[mod_value][element])
    print(f"{orig[element]}{line}")
    line = ""


