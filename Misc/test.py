import itertools

import numpy as np
import numpy as np
from numpy.lib.format import open_memmap
import pandas as pd
import os
from datetime import datetime
from common.utils import load_data, mape, rmse
from time import time


import tempfile, numpy as np, pandas as pd

def todate(dat):
    return datetime.fromtimestamp(dat/ 1000).strftime('%Y-%m-%d %H:%M:%S')

def create_ndarray_from_csv(csv_file): # load csv file to int8 normal/memmap ndarray
    start = time()
    df_int8 = pd.read_csv(csv_file, dtype=np.float32, header=1)
    arr_int8 = df_int8.values
    del df_int8

    memmap_file = tempfile.NamedTemporaryFile(prefix='ndarray-memmap', suffix='.npy')
    np.save(memmap_file.name, arr_int8)
    del arr_int8

    arr_mm_int8 = np.load(memmap_file.name, mmap_mode='r')
    iterator = map(todate, arr_mm_int8[:,0])
    #arr_mm_int8[:,0] = datetime.fromtimestamp(arr_mm_int8[:,0]/ 1000).strftime('%Y-%m-%d %H:%M:%S')
    print(len(list(iterator)))
    print(f"{(time()-start)} seconds")
    return arr_mm_int8

create_ndarray_from_csv('/home/sachin/Thesis/data/Value.csv')

start = time()

vibration = load_data('/home/sachin/Thesis/data')[['vibration']]

print(f"{(time()-start)} seconds")



# train_start_dt = '2021-08-27 06:00:00'
# test_start_dt = '2021-08-27 12:00:00'
# test_end_dt = '2021-08-27 13:00:00'
#
# train_start_dt = pd.Timestamp(train_start_dt) + pd.DateOffset(hours=18)
#
# print(train_start_dt)