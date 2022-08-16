import random
from common.utils import load_data
import pandas as pd
from time import time
import numpy as np
from copy import deepcopy

y_actual = load_data('/home/sachin/Thesis/Anomaly')[['y_actual']]
y_pred = load_data('/home/sachin/Thesis/Anomaly')[['y_pred']]


test_start_dt = '2021-05-27 18:00:00'
test_end_dt = '2021-05-27 19:00:00'


def get_rmse_and_variance(current_hour):
    minus_six = str(pd.Timestamp(current_hour) - pd.DateOffset(hours=6))
    past_six_actual = y_actual.copy()[(y_actual.index >= minus_six) & (y_actual.index < current_hour)][['y_actual']]
    past_six_pred = y_pred.copy()[(y_actual.index >= minus_six) & (y_actual.index < current_hour)][['y_pred']]
    pe = []
    for actual, pred in zip(past_six_actual["y_actual"].tolist(), past_six_pred["y_pred"].tolist()):
        pe.append(abs((actual-pred)/actual,))
    pe_mean = np.array(pe).mean()
    pe_std = np.array(pe).std()
    return pe_mean, pe_std

train = y_actual.copy()[(y_actual.index >= test_start_dt) & (y_actual.index < test_end_dt)][['y_actual']]
train_pred = y_pred.copy()[(y_actual.index >= test_start_dt) & (y_actual.index < test_end_dt)][['y_pred']]
selects = random.sample(range(len(train)), round(len(train)/2))

def modifyList(pred, mod_value):
    for element in selects:
        pred[element] = mod_value * pred[element]
        pred[element] = round(pred[element], 4)
        #pred[ind] = pred[selects]
    return pred


pred = train_pred["y_pred"].to_list()

orig = train["y_actual"].to_list()

mod = train["y_actual"].to_list()

mod_values = []


# i =2.5
#
# while i <= 40:
#     mod_values.append(i)
#     i = i*2

mods = np.arange(1.0, 3.0, 0.1)
mod_values = [round(n, 2) for n in mods]

#print(mod_values)

mod_list = {}

for mod_value in mod_values:
    mod = deepcopy(orig)
    mod_return = modifyList(mod, mod_value)
    mod_list[mod_value] = deepcopy(mod_return)



diff_list = []
same_list = []

h_values = np.arange(1.0, 8.5, 0.5)
hs = [round(h, 2) for h in h_values]


rms, std = get_rmse_and_variance(test_start_dt)
rms = round(rms, 4)
std = round(std, 4)

# print(rms, std)

# for key in mod_list:
#     print(f"{key} : {mod_list[key]}")

value_dict = {}
h_dict = {}
h_dict_main = {}
fpr_list = []
fnr_list = []
index = []



for key in mod_list:
    for h in hs:
        #mod_date = test_start_dt
        start = time()
        for element in range(0, len(mod)):
            #rms, std = get_rmse_and_variance(mod_date)
            diff = abs(mod_list[key][element] - pred[element])
            if not rms-h*std < diff < rms+h*std:
                if element not in selects:
                    diff_list.append(element)
                else:
                    same_list.append(element)
            #mod_date = str(pd.Timestamp(mod_date) + pd.DateOffset(seconds=1))
        tt = (time() - start)
        print(f"mod_value: {key}, TT: {round(tt)}, h: {h}, FP: {len(diff_list)}, FN: {(len(selects)-len(same_list))},"
              f" FPR: {len(diff_list)/(len(selects))}, FNR: {(len(selects)-len(same_list))/(len(selects))}")
        FPR = len(diff_list)/(len(selects))
        FNR = (len(selects)-len(same_list))/len(selects)
        index.append(round(h, 1))
        fpr_list.append(deepcopy(FPR))
        fnr_list.append(deepcopy(FNR))
        h_dict_main["index"] = deepcopy(index)
        h_dict_main["FPR"] = deepcopy(fpr_list)
        h_dict_main["FNR"] = deepcopy(fnr_list)
        diff_list = []
        same_list = []
    fpr_list = []
    fnr_list = []
    index = []
    value_dict[key] = deepcopy(h_dict_main)
    print("\n")

# for key in value_dict:
#     print(value_dict[key])