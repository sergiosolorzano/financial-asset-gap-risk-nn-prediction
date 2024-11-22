#!/usr/bin/env python
import pandas as pd
import matplotlib.pyplot as plt

# Data dictionary
data = {
    "Eval Stock": ["OZK", "PWBK", "SIVBQ", "ZION", "FITB", "KEY", "RF"],
    "DTW Distance": [1261.1, 999.9, 5443, 628.8, 632.7, 514, 2020],
    "Input Images SSIM": [0.5620, 0.0565, 0.3778, 0.5414, 0.72, 0.8, 0.6275],
    "Feature Image Conv2d SSIM Inputs Training-Vs-Eval": [0.5858, 0.2229, 0.4032, 0.5541, 0.6532, 0.7382, 0.6176],
    "Feature Image FC SSIM Inputs Training-Vs-Eval": [0.3172, 0.1444, 0.3152, 0.4291, 0.5092, 0.5067, 0.4633],
    "Evaluation Threshold Accuracy 1 dp (%)": [50.0, 16.66, 5.55, 38.88, 58.33, 44.44, 61.11],
    "Evaluation MAE": [0.1559, 0.2888, 0.2775, 0.1833, 0.1233, 0.1426, 0.09514],
    "Evaluation R^2": [0.3140, -0.0830, 0.0273, 0.2563, 0.6964, 0.5724, 0.6998]
}

# Convert the data into a DataFrame
df = pd.DataFrame(data)

# Sort the entire DataFrame by "Input Images SSIM"
df = df.sort_values(by='Input Images SSIM')

# Plotting the graph with lines and markers
plt.figure(figsize=(10, 6))
plt.plot(df["Input Images SSIM"], df["Feature Image Conv2d SSIM Inputs Training-Vs-Eval"], marker='o', label="SSIM Train_Input-Conv")
plt.plot(df["Input Images SSIM"], df["Feature Image FC SSIM Inputs Training-Vs-Eval"], marker='o', label="SSIM Train_Input-FC")
plt.plot(df["Input Images SSIM"], df["Evaluation Threshold Accuracy 1 dp (%)"] / 100, marker='o', label="Eval Threshold 1 dp")
plt.plot(df["Input Images SSIM"], df["Evaluation MAE"], marker='o', label="Evaluation MAE")
plt.plot(df["Input Images SSIM"], df["Evaluation R^2"], marker='o', label="Evaluation R^2")

# Adding labels and title
plt.xlabel("SSIM")
plt.ylabel("Metrics")
plt.title("Comparison of Metrics vs SSIM")
plt.legend()
plt.grid(visible=True, linestyle='--', alpha=0.5)

plt.show()


