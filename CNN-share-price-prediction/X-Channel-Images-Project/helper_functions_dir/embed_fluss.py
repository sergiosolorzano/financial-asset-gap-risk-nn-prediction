#install: conda install -c conda-forge cuda-nvcc cuda-nvrtc "cuda-version>=12.1"
import os
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import stumpy
import yfinance as yf
from sklearn.preprocessing import StandardScaler,MinMaxScaler

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

print(dataset_df)

scaler = MinMaxScaler(Parameters.min_max_scaler_feature_range)
print("BEFORE SCALER")
#scaled_data_f32 = Parameters.scaler.fit_transform(dataset_df['Close'].values.reshape(-1, 1))
vals = dataset_df['Close'].values.reshape(-1, 1)
print("type",type(vals))
scaled_data = scaler.fit_transform(vals).reshape(-1,)
print("AFTER SCALER len",len(scaled_data))

#Matrix profile and window size
window_size = 32
# matrix_profile 4 index result meaning:
#[distance, index_of_nearest_neighbor, motif_index, motif_length]
# Distance: Euclidean distance between the current subsequence and its nearest neighbor in the time series
# Index: The index of the nearest neighbor subsequence in the time series
# Motif Index: The index of the current subsequence in the time series
# Motif Length: The length of the subsequence

print("len reshaped",len(scaled_data))
matrix_profile = stumpy.gpu_stump(scaled_data, m=window_size)
print("matrix values",matrix_profile.P_,"matrix indices", matrix_profile.I_)
matrix_profile_df = pd.DataFrame(matrix_profile, columns=['profile', 'profile index', 'left profile index', 'right profile index'])
print(matrix_profile_df)                                                          
#identify change points with FLUSS
#spaced out segments by min excl_factor* window_size
correct_arc_curve, regime_locations = stumpy.fluss(matrix_profile[:, 1], L=window_size, n_regimes=5, excl_factor=5)
print(correct_arc_curve)
print("---")
print(regime_locations)
#Find the change points (discontinuities in semantic state)
change_points = np.argwhere(np.diff(correct_arc_curve['T']) < 0).flatten()

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
