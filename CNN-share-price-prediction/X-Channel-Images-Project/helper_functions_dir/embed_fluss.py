#install: conda install -c conda-forge cuda-nvcc cuda-nvrtc "cuda-version>=12.1"
import os
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import stumpy
import yfinance as yf

import importlib as importlib
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
from parameters import Parameters

from numba import cuda

if cuda.is_available():
    print("Found cuda for numba")
else:
    print("no cuda for numba")


dataset_df = yf.download(Parameters.train_stock_ticker, start=Parameters.start_date, end=Parameters.end_date, interval='1d')
dataset_df = dataset_df.dropna()

dataset_df = dataset_df.reset_index()

#reorder to split the data to train and test
desired_order = ['Date','Open', 'Close', 'High', 'Low']
if 'Date' in dataset_df.columns:
    dataset_df = dataset_df[desired_order]
else:
    print("Column 'Date' is missing.")

dataset_df = dataset_df.set_index('Date')

#Matrix profile and window size
window_size = 32
matrix_profile = stumpy.gpu_stump(dataset_df['Close'], m=window_size)

print(matrix_profile)

#identify change points with FLUSS
#spaced out segments by min excl_factor* window_size
flux = stumpy.fluss(matrix_profile[:, 1], L=window_size, n_regimes=5, excl_factor=5)

# Find the change points (discontinuities in semantic state)
change_points = np.argwhere(np.diff(flux['T']) < 0).flatten()

# Visualize the change points on the time series
plt.figure(figsize=(12, 6))
plt.plot(dataset_df['Date'], dataset_df['Price'], label='Share Price')
plt.scatter(dataset_df['Date'].iloc[change_points], dataset_df['Price'].iloc[change_points], color='red', label='Change Points')
plt.title('Time Series with Change Points Detected using FLUSS')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

print("Detected Change Points (Index):", change_points)
print("Detected Change Points (Dates):", dataset_df['Date'].iloc[change_points].values)
