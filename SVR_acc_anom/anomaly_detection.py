
from common.utils import load_data
import numpy as np
from Anomaly.ModifyList import ModifyList
from Anomaly.Detection import Detection
from matplotlib import pyplot as plt
import pandas as pd
import os

def load_data(data_dir):
    energy = pd.read_csv(os.path.join(data_dir, 'acc.out'))
    return energy


def get_pe_mean_and_std(actual_data, predicted_data):
    percentage_errors = [abs((actual - predicted) / actual) for actual, predicted in zip(actual_data, predicted_data)]
    pe_mean = round(np.array(percentage_errors).mean(), 5)
    pe_std = round(np.array(percentage_errors).std(), 5)
    return pe_mean, pe_std

def get__mean_and_std(actual_data, predicted_data):
    pe_mean = round(np.array(predicted_data).mean(), 5)
    pe_std = round(np.array(predicted_data).std(), 5)
    return pe_mean, pe_std


mean = 0
std = 0

y_actual = load_data('/home/sachin/Thesis/SVR_acc_anom')[['test']]
y_pred = load_data('/home/sachin/Thesis/SVR_acc_anom')[['pred']]

check_start_dt_pe = 0
check_start_dt = 500
check_end_dt = 600

actual_values = y_actual.copy()[(y_actual.index >= check_start_dt) & (y_actual.index < check_end_dt)][['test']]
predicted_values = y_pred.copy()[(y_pred.index >= check_start_dt) & (y_pred.index < check_end_dt)][['pred']]

values_for_mean = y_actual.copy()[(y_actual.index >= check_start_dt_pe) & (y_actual.index < check_start_dt)][['test']]
pred_for_mean = y_pred.copy()[(y_pred.index >= check_start_dt_pe) & (y_pred.index < check_start_dt)][['pred']]

mean, std = get_pe_mean_and_std(values_for_mean["test"].to_list(), pred_for_mean["pred"].to_list())


d_mean, d_std = get__mean_and_std(values_for_mean["test"].to_list(), pred_for_mean["pred"].to_list())
# print(d_std/d_mean*100)


modification_values = [round(n, 1) for n in np.arange(1.2, 3.1, 0.1)]
h_values = [round(n, 2) for n in np.arange(1, 8.5, 0.5)]
# h_values = [round(n, 2) for n in np.arange(3, 5, 0.05)]
# h_values = [round(n, 2) for n in np.arange(4.4, 4.6, 0.05)]
# h_values = [round(n, 2) for n in np.arange(4.47, 4.52, 0.01)]
# h_values = [4]

FNR = []
FPR = []
FNR_mean = []
FPR_mean = []

detect_object_list = []
h_value_result_FNR = {}
h_value_result_FPR = {}

for _ in h_values:
    h_value_result_FNR[_] = []
    h_value_result_FPR[_] = []



for mod_value in modification_values:
    for h in h_values:
        mod = ModifyList(actual_values["test"].to_list(), mod_value, 75)
        det = Detection(predicted_values["pred"].to_list(), mod.modified_list, mod.selects, mean, std, h)

        print(f"Anomaly Modification: {mod_value}  H: {h}  TPR: {round(det.sensitivity,2)}  "
              f"TNR: {round(det.specificity,2)}  FPR: {round(1 - det.specificity, 2)}  "
              f"FNR: {round(1 - det.sensitivity, 2)}")
        h_value_result_FNR[h] += [round(1 - det.sensitivity, 2)]
        h_value_result_FPR[h] += [round(1 - det.specificity, 2)]

    # plt.plot(h_values, FNR,
    #          h_values, FPR)
    # plt.show()
    print("\n")

for _ in h_values:
    #print(f"{_} : {h_value_result_FNR[_]} {h_value_result_FPR[_]}")
    FNR_mean.append(np.array(h_value_result_FNR[_]).mean()*100)
    FPR_mean.append(np.array(h_value_result_FPR[_]).mean()*100)
    print(f"{_} {h_value_result_FNR[_]} {h_value_result_FPR[_]}")


for _ in range(len(h_values)):
    print(f"{h_values[_]} {FNR_mean[_]} {FPR_mean[_]}")

cost = FNR_mean[0]*3000 + FPR_mean[0]*30

print(f"cost: {cost}")

print(h_values, "\n", FNR_mean, "\n", FPR_mean, "\n",)

plt.figure(figsize=(10,4))
plt.plot(h_values, FNR_mean, color='green', linewidth=2.0, alpha = 0.6)
plt.plot(h_values, FPR_mean, color='blue', linewidth=2.0, alpha = 0.6)
plt.legend(['FNR', 'FPR'], loc="upper right")
plt.title("FNR vs FPR")
plt.xlabel('h Values')
plt.ylabel('Percentage')
#
# plt.savefig("FNR vs FPR{fine} .eps", format="eps", dpi=1200)
# plt.savefig("FNR vs FPR{fine} .jpg", format="jpg", dpi=1200)

plt.show()