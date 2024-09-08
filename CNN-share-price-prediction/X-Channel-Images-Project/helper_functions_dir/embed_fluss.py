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

dataset_df['Index_Number'] = dataset_df.index
log_diffs = np.diff(np.log(dataset_df['Close']))
log_diffs = np.insert(log_diffs, 0, np.nan)
dataset_df['LogDiff'] = log_diffs
dataset_df = dataset_df.set_index('Date')

pd.set_option("display.max_rows", 1000)
pd.set_option("display.expand_frame_repr", True)
pd.set_option('display.width', 1000)
pd.set_option("display.max_colwidth", 1000)

print(dataset_df)

#Matrix profile and window size
window_size = 5
# matrix_profile 4 index result meaning:
#[distance, index_of_nearest_neighbor, motif_index, motif_length]
# Distance: Euclidean distance between the current subsequence and its nearest neighbor in the time series
# Index: The index of the nearest neighbor subsequence in the time series
# Motif Index: The index of the current subsequence in the time series
# Motif Length: The length of the subsequence

#sump z-normalizes
matrix_profile = stumpy.gpu_stump(dataset_df['Close'], m=window_size)
#print("matrix values",matrix_profile.P_,"matrix indices", matrix_profile.I_)
motif_idx = np.argsort(matrix_profile[:, 0])[0]
print(f"The motif is located at index {motif_idx}")
nearest_neighbor_idx = matrix_profile[motif_idx, 1]
print(f"The nearest neighbor is located at index {nearest_neighbor_idx}")
print("motif global minima",matrix_profile[:, 0].min())
print("len matrix profile",len(matrix_profile))


filtered_indices = np.where((matrix_profile[:, 0] <= 0.005) & (matrix_profile[:, 0] >= 0.0001))[0]

max_index = np.argmax(dataset_df['LogDiff'].values)
max_value = dataset_df['LogDiff'].iloc[max_index]
print("max value",max_value)

# Find the index of the minimum value in the 'LogDiff' column
min_index = np.argmin(dataset_df['LogDiff'].values)
min_value = dataset_df['LogDiff'].iloc[min_index]
print("min value",min_index)

for idx in filtered_indices:
    print(f"Index: {idx}, Value: {idx} nearst neighbor {matrix_profile[idx, 1]} nearest left neighbor {matrix_profile[idx, 2]} log price change {dataset_df['LogDiff'].iloc[matrix_profile[idx, 2]]}")

if len(filtered_indices) > 0:
    # Choose the first index within the range (you can adjust this based on your needs)
    adjusted_motif_idx = filtered_indices[0]
    print(f"Adjusted Motif index: {adjusted_motif_idx} nearst neighbor {matrix_profile[adjusted_motif_idx, 1]} nearest left neighbor ",matrix_profile[adjusted_motif_idx, 2])
else:
    print("No adjusted Motif index")

fig, axs = plt.subplots(2, sharex=True, gridspec_kw={'hspace': 0})
plt.suptitle('Motif (Pattern) Discovery', fontsize='30')
axs[0].plot(dataset_df['Close'].values)
axs[0].set_ylabel('Close', fontsize='20')
rect = plt.Rectangle((motif_idx, 0), window_size, np.max(dataset_df['Close']), facecolor='lightgrey')
axs[0].add_patch(rect)
rect = plt.Rectangle((nearest_neighbor_idx, 0), window_size, np.max(dataset_df['Close']), facecolor='lightgrey')
axs[0].add_patch(rect)
axs[1].set_xlabel('Time', fontsize ='20')
axs[1].set_ylabel('Matrix Profile', fontsize='20')
axs[1].axvline(x=motif_idx, linestyle="dashed")
axs[1].axvline(x=nearest_neighbor_idx, linestyle="dashed")
axs[1].plot(matrix_profile[:, 0])
#plt.show()

matrix_profile_df = pd.DataFrame(matrix_profile, columns=['profile', 'profile index', 'left profile index', 'right profile index'])
print(matrix_profile_df)                                                          
conditions = [matrix_profile_df.iloc[:, 3] == value for value in [316, 317, 318, 319, 320]]
filtered_df = matrix_profile_df[(conditions[0]) | (conditions[1]) | (conditions[2]) | (conditions[3]) | (conditions[4])]
index = filtered_df.index.tolist()
print("index",index)

#identify change points with FLUSS
#spaced out segments by min excl_factor* window_size
correct_arc_curve, regime_locations = stumpy.fluss(matrix_profile[:, 1], L=window_size, n_regimes=5, excl_factor=5)
# print(correct_arc_curve)
# print("---")
# print(regime_locations)
#Find the change points (discontinuities in semantic state)
# change_points = np.argwhere(np.diff(correct_arc_curve['T']) < 0).flatten()

# # Visualize the change points on the time series
# plt.figure(figsize=(12, 6))
# plt.plot(dataset_df['Date'], dataset_df['Price'], label='Share Price')
# plt.scatter(dataset_df['Date'].iloc[change_points], dataset_df['Price'].iloc[change_points], color='red', label='Change Points')
# plt.title('Time Series with Change Points Detected using FLUSS')
# plt.xlabel('Date')
# plt.ylabel('Price')
# plt.legend()
#plt.show()

# print("Detected Change Points (Index):", change_points)
# print("Detected Change Points (Dates):", dataset_df['Date'].iloc[change_points].values)
