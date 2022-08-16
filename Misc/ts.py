#!/usr/bin/python3.6

import os
import matplotlib.pyplot as plt
import common.utils

data_dir = '../data'
energy = common.utils.load_data(data_dir)[['vibration']]
#print(len(energy.index))
#energy.head()

#Plotting the whle data
# energy.plot(y='vibration', subplots=True, figsize=(15, 8), fontsize=12)
# plt.xlabel('timestamp', fontsize=12)
# plt.ylabel('vibration', fontsize=12)
# plt.show()

train_start_dt = '2021-04-23 09:43:24'
test_start_dt = '2021-04-24 9:43:24'

energy[train_start_dt:test_start_dt].plot(y='vibration', subplots=True, figsize=(15, 8), fontsize=12)
plt.xlabel('timestamp', fontsize=12)
plt.ylabel('vibration', fontsize=12)
plt.show()