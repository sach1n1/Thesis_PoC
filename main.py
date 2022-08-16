import os
import matplotlib.pyplot as plt
from common.utils import load_data

data_dir = './data'

energy = load_data('data_dir')[['load']]
energy.head()

energy.plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
plt.xlabel('timestamp', fontsize=12)
plt.ylabel('load', fontsize=12)
plt.show()