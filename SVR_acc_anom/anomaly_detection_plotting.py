from common.utils import load_data
from Anomaly.ModifyList import ModifyList
from matplotlib import pyplot as plt
import pandas as pd
import os

def load_data(data_dir):
    energy = pd.read_csv(os.path.join(data_dir, 'acc.out'))
    return energy

y_actual = load_data('/home/sachin/Thesis/SVR_acc_anom')[['test']]
y_pred = load_data('/home/sachin/Thesis/SVR_acc_anom')[['pred']]



check_start_dt_pe = 140
check_start_dt = 200
check_end_dt = 260

actual_values = y_actual.copy()[(y_actual.index >= check_start_dt) & (y_actual.index < check_end_dt)][['test']]
predicted_values = y_pred.copy()[(y_pred.index >= check_start_dt) & (y_pred.index < check_end_dt)][['pred']]

print(predicted_values)

modification_values = [1.2]
percentage_of_mods = [25, 50, 75]

for mod_value in modification_values:
    for percent in percentage_of_mods:
        mod = ModifyList(actual_values["test"].to_list(), mod_value, percent)
        actual_values["pred"] = mod.modified_list
        plt.plot(actual_values["test"], color='green', linewidth=2.0, alpha=0.6)
        plt.plot(actual_values["pred"], color='blue', linestyle='dotted', alpha=0.6)
        plt.plot(predicted_values, color='black', linestyle='dashed', alpha=0.6)
        plt.title(f"Actual Values vs Modified with {percent}% Anomalies")
        plt.xlabel("Time")
        plt.ylabel("Vibration Values")
        plt.legend(['Actual Values', 'Actual Values modified with Anomalies'], loc="upper right")
        plt.savefig(f"plots/Actual Values vs Modified with {percent} Anomalies.eps", format="eps", dpi=1200)
        plt.savefig(f"plots/Actual Values vs Modified with {percent} Anomalies.jpg", format="jpg", dpi=1200)
        plt.show()
