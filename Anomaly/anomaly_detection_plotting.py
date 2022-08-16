
from common.utils import load_data
import numpy as np
from ModifyList import ModifyList
from Detection import Detection
from matplotlib import pyplot as plt


def get_pe_mean_and_std(actual_data, predicted_data):
    percentage_errors = [abs((actual - predicted) / actual) for actual, predicted in zip(actual_data, predicted_data)]
    pe_mean = round(np.array(percentage_errors).mean(), 5)
    pe_std = round(np.array(percentage_errors).std(), 5)
    return pe_mean, pe_std

def get__mean_and_std(actual_data, predicted_data):
    pe_mean = round(np.array(predicted_data).mean(), 5)
    pe_std = round(np.array(predicted_data).std(), 5)
    return pe_mean, pe_std


y_actual = load_data('/home/sachin/Thesis/Anomaly')[['y_actual']]
y_pred = load_data('/home/sachin/Thesis/Anomaly')[['y_pred']]

check_start_dt_pe = '2021-05-27 14:00:00'
check_start_dt = '2021-05-27 20:00:00'
check_end_dt = '2021-05-27 20:01:00'

actual_values = y_actual.copy()[(y_actual.index >= check_start_dt) & (y_actual.index < check_end_dt)][['y_actual']]
predicted_values = y_pred.copy()[(y_pred.index >= check_start_dt) & (y_pred.index < check_end_dt)][['y_pred']]

modification_values = [1.5]
percentage_of_mods = [25, 50, 75]



for mod_value in modification_values:
    for percent in percentage_of_mods:
        mod = ModifyList(actual_values["y_actual"].to_list(), mod_value, percent)
        actual_values["y_pred"] = mod.modified_list
        plt.plot(actual_values["y_actual"], color='green', linewidth=2.0, alpha=0.6)
        plt.plot(actual_values["y_pred"], color='blue', linestyle='dotted', alpha=0.6)
        plt.legend(['Actual Values', 'Actual Values modified with Anomalies'], loc="upper right")
        actual_values.plot()
        plt.show()
