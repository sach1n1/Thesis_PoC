import numpy as np
from ModifyList import ModifyList
from Detection import Detection
from matplotlib import pyplot as plt
import pandas as pd
import os


def load_data(data_dir):
    energy = pd.read_csv(os.path.join(data_dir, 'OutActualSSForecast.csv')
                         , parse_dates=['timestamp'])
    energy = energy.drop_duplicates(subset="timestamp", keep='first')
    energy.index = energy['timestamp']
    energy = energy.reindex(pd.date_range(min(energy['timestamp']),
                                          max(energy['timestamp']),
                                          freq='S'))
    energy = energy.drop('timestamp', axis=1)
    energy = energy.interpolate()
    return energy


def get_pe_mean_and_std(actual_data, predicted_data):
    percentage_errors = [abs((actual - predicted) / actual) for actual, predicted in zip(actual_data, predicted_data)]
    pe_mean = round(np.array(percentage_errors).mean(), 5)
    pe_std = round(np.array(percentage_errors).std(), 5)
    return pe_mean, pe_std


p = 75

mean = 0
std = 0

y_actual = load_data('../data/')[['y_actual']]
y_pred = load_data('../data/')[['y_pred']]

check_start_dt_pe = '2021-05-27 14:00:00'
check_start_dt = '2021-05-27 20:00:00'
check_end_dt = '2021-05-27 21:00:00'

actual_values = y_actual.copy()[(y_actual.index >= check_start_dt) & (y_actual.index < check_end_dt)][['y_actual']]
predicted_values = y_pred.copy()[(y_pred.index >= check_start_dt) & (y_pred.index < check_end_dt)][['y_pred']]

values_for_mean = y_actual.copy()[(y_actual.index >= check_start_dt_pe) & (y_actual.index < check_start_dt)][['y_actual']]
pred_for_mean = y_pred.copy()[(y_pred.index >= check_start_dt_pe) & (y_pred.index < check_start_dt)][['y_pred']]

mean, std = get_pe_mean_and_std(values_for_mean["y_actual"].to_list(), pred_for_mean["y_pred"].to_list())




modification_values = [round(n, 1) for n in np.arange(1.3, 1.9, 0.1)]
# h_values = [round(n, 2) for n in np.arange(1, 8.5, 0.5)]
# h_values = [round(n, 2) for n in np.arange(3.5, 4.05, 0.05)]
# h_values = [round(n, 2) for n in np.arange(3.9, 4, 0.01)]
# h_values = [round(n, 2) for n in np.arange(4.47, 4.52, 0.01)]
h_values = [3.9]

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
        mod = ModifyList(actual_values["y_actual"].to_list(), mod_value, p)
        det = Detection(predicted_values["y_pred"].to_list(), mod.modified_list, mod.selects, mean, std, h)

        # print(f"Anomaly Modification: {mod_value}  H: {h}  TPR: {round(det.sensitivity,2)}  "
        #       f"TNR: {round(det.specificity,2)}  FPR: {round(1 - det.specificity, 2)}  "
        #       f"FNR: {round(1 - det.sensitivity, 2)}")
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

print(h_values, "\n", round(FNR_mean[0],2), "\n", round(FPR_mean[0],2), "\n",)

# plt.figure(figsize=(10,4))
# plt.plot(h_values, FNR_mean, color='green', linewidth=2.0, alpha = 0.6)
# plt.plot(h_values, FPR_mean, color='blue', linewidth=2.0, alpha = 0.6)
# plt.legend(['FNR', 'FPR'], loc="upper right")
# # plt.title("FNR vs FPR")
# plt.xlabel('h Values')
# plt.ylabel('Percentage')
#
# plt.savefig(f"plots/FNR vs FPR_{p}_fine1.jpg", format="jpg", dpi=1200)
#
# plt.show()