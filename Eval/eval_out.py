import ast
import pandas as pd
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt

import os


Path = "/home/sachin/Thesis/Eval/"
file_list = ['fb_hour1.txt', 'fb_hour6.txt', 'fb_hour12.txt', 'fb_hour24.txt',
             'SVR_hour1.txt', 'SVR_hour6.txt', 'SVR_hour12.txt', 'SVR_hour24.txt']

values = {}

for file in file_list:
    f = open(f"/home/sachin/Thesis/Eval/{file}", 'r')
    data = f.read()
    #key_ID = file.split("_")[0] + "_" + file.split("_")[1].replace(".txt", "")
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

predictor = ["FB", "SVR"]
duration = ["1hr", "6hrs", "12hrs", "24hrs"]

df = pd.DataFrame(data_list, columns=['FB MAPE 1hr', 'FB RMSE 1hr', 'FB TT 1hr',
                                      'FB MAPE 6hrs', 'FB RMSE 6hrs', 'FB TT 6hrs',
                                      'FB MAPE 12hrs', 'FB RMSE 12hrs', 'FB TT 12hrs',
                                      'FB MAPE 24hrs', 'FB RMSE 24hrs', 'FB TT 24hrs',
                                      'SVR MAPE 1hr', 'SVR RMSE 1hr', 'SVR TT 1hr',
                                      'SVR MAPE 6hrs', 'SVR RMSE 6hrs', 'SVR TT 6hrs',
                                      'SVR MAPE 12hrs', 'SVR RMSE 12hrs', 'SVR TT 12hrs',
                                      'SVR MAPE 24hrs', 'SVR RMSE 24hrs', 'SVR TT 24hrs'])
def divide_100(x):
    return x/100


df['SVR RMSE 1hr'] = df['SVR RMSE 1hr'].apply(divide_100)
df['SVR RMSE 6hrs'] = df['SVR RMSE 6hrs'].apply(divide_100)
df['SVR RMSE 12hrs'] = df['SVR RMSE 12hrs'].apply(divide_100)
df['SVR RMSE 24hrs'] = df['SVR RMSE 24hrs'].apply(divide_100)
# df.reset_index().plot(x='index', title='MAPE over time range for FBProphet.', xlabel='Time Slot Index', ylabel='MAPE %',
#                       y=['FB MAPE 1hr', 'FB MAPE 6hrs', 'FB MAPE 12hrs', 'FB MAPE 24hrs'],
#                       kind='line', style='.-')
# df.reset_index().plot(x='index', title='RMSE over time range for FBProphet.', xlabel='Time Slot Index', ylabel='RMSE %',
#                       y=['FB RMSE 1hr', 'FB RMSE 6hrs', 'FB RMSE 12hrs', 'FB RMSE 24hrs'],
#                       kind='line', style='.-')
ax = df.reset_index().plot(x='index', title='RMSE over time range for SVR.', xlabel='Time Slot Index', ylabel='RMSE',
                      y=['SVR RMSE 1hr', 'SVR RMSE 6hrs', 'SVR RMSE 12hrs', 'SVR RMSE 24hrs'],
                      kind='line', style='.-')
plt.axhline(y=0.1, color='r', linestyle='--')
L=plt.legend()
L.get_texts()[0].set_text('RMSE 1 hr')
L.get_texts()[1].set_text('RMSE 6 hrs')
L.get_texts()[2].set_text('RMSE 12 hrs')
L.get_texts()[3].set_text('RMSE 24 hrs')
plt.savefig("RMSE_SVR.eps", format="eps", dpi=1200)
plt.show()
# df.reset_index().plot(x='index', title='MAPE over time range for SVR.', xlabel='Time Slot Index', ylabel='MAPE %',
#                       y=['SVR MAPE 1hr', 'SVR MAPE 6hrs', 'SVR MAPE 12hrs', 'SVR MAPE 24hrs'],
#                       kind='line', style='.-')
df.reset_index().plot(x='index', title='SVR Time Taken for training.', xlabel='Time Slot Index', ylabel='Time (minutes)',
                      y=['SVR TT 1hr', 'SVR TT 6hrs', 'SVR TT 12hrs', 'SVR TT 24hrs'],
                      kind='line', style='.-')
plt.savefig("SVR_TT.eps", format="eps", dpi=1200)
SVR_opt = {}
single = {}

durations = ['1hr', '6hrs', '12hrs', '24hrs']

#print(df[f"SVR RMSE 1hr"].mean())



for duration in durations:
    single["RMSE MEAN"] = round(df[f"SVR RMSE {duration}"].mean(), 2)
    single["MAPE MEAN"] = round(df[f"SVR MAPE {duration}"].mean(), 2)
    single["TT MEAN"] = round(df[f"SVR TT {duration}"].mean(), 2)
    single["RMSE opt"] = round(0.6 * single["RMSE MEAN"] + 0.4*single["TT MEAN"], 2)
    single["MAPE opt"] = round(0.6 * single["MAPE MEAN"] + 0.4 * single["TT MEAN"], 2)
    SVR_opt[f"SVR {duration}"] = single
    single = {}
#
# for duration in durations:
#     single["RMSE MEAN"] = round(df[f"FB RMSE {duration}"].mean(), 2)
#     single["MAPE MEAN"] = round(df[f"FB MAPE {duration}"].mean(), 2)
#     single["TT MEAN"] = round(df[f"FB TT {duration}"].mean(), 2)
#     single["RMSE opt"] = round(0.6 * single["RMSE MEAN"] + 0.4*single["TT MEAN"], 2)
#     single["MAPE opt"] = round(0.6 * single["MAPE MEAN"] + 0.4 * single["TT MEAN"], 2)
#     SVR_opt[f"FB {duration}"] = single
#     single = {}
#
for duration in SVR_opt:
    print(duration, SVR_opt[duration])

# plt.savefig("thenga.png", dpi=1200)

# plt.show()
