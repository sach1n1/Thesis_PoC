import ast
import pandas as pd
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt


Path = "/home/sachin/Thesis/Eval/"
file_list = ['hour1CV.txt', 'hour6CV.txt', 'hour12CV.txt', 'hour24CV.txt']

values = {}

for file in file_list:
    f = open(f"/home/sachin/Thesis/Eval/{file}", 'r')
    data = f.read()
    values[file] = ast.literal_eval(data)
    f.close()

item_list = []
data_list = []
number = 0

features = ["MAPE", "RMSE", "Time Taken"]

for file in file_list:
    for feature in features:
        for key in range(1, 31):
            item_list.append(values[file][str(key)][feature])
        data_list.append(deepcopy(item_list))
        item_list.clear()


numpy_array = np.array(data_list)
transpose = numpy_array.T
data_list = transpose.tolist()

predictor = ["SVR"]
duration = ["1hr", "6hrs", "12hrs", "24hrs"]

df = pd.DataFrame(data_list, columns=['SVR MAPE 1hr', 'SVR RMSE 1hr', 'SVR TT 1hr',
                                      'SVR MAPE 6hrs', 'SVR RMSE 6hrs', 'SVR TT 6hrs',
                                      'SVR MAPE 12hrs', 'SVR RMSE 12hrs', 'SVR TT 12hrs',
                                      'SVR MAPE 24hrs', 'SVR RMSE 24hrs', 'SVR TT 24hrs'])


ax = df.reset_index().plot(x='index', xlabel='Time Slot', ylabel='MAPE (%)',
                      y=['SVR MAPE 1hr', 'SVR MAPE 6hrs', 'SVR MAPE 12hrs', 'SVR MAPE 24hrs'],
                      kind='line', style='.-')
plt.axhline(y=6.5, color='r', linestyle='--')
L=plt.legend()
L.get_texts()[0].set_text('MAPE 1 hr')
L.get_texts()[1].set_text('MAPE 6 hrs')
L.get_texts()[2].set_text('MAPE 12 hrs')
L.get_texts()[3].set_text('MAPE 24 hrs')
plt.savefig("plots/MAPE_SVR.jpg", format="jpg", dpi=1200)
plt.show()

ax = df.reset_index().plot(x='index', xlabel='Time Slot', ylabel='RMSE',
                      y=['SVR RMSE 1hr', 'SVR RMSE 6hrs', 'SVR RMSE 12hrs', 'SVR RMSE 24hrs'],
                      kind='line', style='.-')
plt.axhline(y=0.1, color='r', linestyle='--')
L=plt.legend()
L.get_texts()[0].set_text('RMSE 1 hr')
L.get_texts()[1].set_text('RMSE 6 hrs')
L.get_texts()[2].set_text('RMSE 12 hrs')
L.get_texts()[3].set_text('RMSE 24 hrs')
plt.savefig("plots/RMSE_SVR.jpg", format="jpg", dpi=1200)
plt.show()



df.reset_index().plot(x='index', xlabel='Time Slot Index', ylabel='Time (minutes)',
                      y=['SVR TT 1hr', 'SVR TT 6hrs', 'SVR TT 12hrs', 'SVR TT 24hrs'],
                      kind='line', style='.-')
plt.savefig("plots/SVR_TT.jpg", format="jpg", dpi=1200)
plt.show()
SVR_opt = {}
single = {}

durations = ['1hr', '6hrs', '12hrs', '24hrs']




for duration in durations:
    single["RMSE MEAN"] = round(df[f"SVR RMSE {duration}"].mean(), 2)
    single["MAPE MEAN"] = round(df[f"SVR MAPE {duration}"].mean(), 2)
    single["TT MEAN"] = round(df[f"SVR TT {duration}"].mean(), 2)
    single["RMSE opt"] = round(0.8 * single["RMSE MEAN"] + 0.2*single["TT MEAN"], 2)
    single["MAPE opt"] = round(0.8 * single["MAPE MEAN"] + 0.2*single["TT MEAN"], 2)
    SVR_opt[f"SVR {duration}"] = single
    single = {}

for duration in SVR_opt:
    print(duration, SVR_opt[duration])

# plt.show()
