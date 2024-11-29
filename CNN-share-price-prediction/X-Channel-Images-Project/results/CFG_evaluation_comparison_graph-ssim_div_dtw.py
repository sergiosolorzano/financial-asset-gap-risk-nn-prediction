#!/usr/bin/env python
import pandas as pd
import matplotlib.pyplot as plt

# Data dictionary
data = {
    "Eval Stock": ["OZK", "PWBK", "SIVBQ", "ZION", "FITB", "KEY", "RF"],
    "DTW Distance To CFG": [1261.1, 999.9, 5443, 628.8, 632.7, 514, 2020],
    "Stock Pair Images SSIM": [0.5620, 0.0565, 0.3778, 0.5414, 0.72, 0.8, 0.6275],
    "SSIM Inputs Training-Vs-Feature Image Conv2d Eval": [0.2290, 0.1134, 0.2096, 0.2377, 0.3029, 0.3318, 0.2618],
    "SSIM Inputs Training-Vs-Feature Image FC Eval": [0.3537, 0.0730, 0.2192, 0.3881, 0.6152, 0.7593, 0.5718],
    "Evaluation Threshold Accuracy 1 dp (%)": [47.22, 25.0, 5.55, 44.44, 61.11, 77.77, 38.88],
    "Evaluation MAE": [0.1341, 0.2781, 0.3090, 0.1521, 0.0962, 0.0684, 0.1505],
    "Evaluation R^2": [0.5069, -0.0373, -0.4345, 0.4482, 0.7820, 0.9119, 0.3529]
}

# Convert the data into a DataFrame
df = pd.DataFrame(data)

# Calculate SSIM/DTW ratio
df["SSIM/DTW"] = df["Stock Pair Images SSIM"] / df["DTW Distance To CFG"]

# Sort the DataFrame by SSIM/DTW
df = df.sort_values(by="SSIM/DTW")

# Plotting
plt.figure(figsize=(12, 7))
plt.plot(df["SSIM/DTW"], df["SSIM Inputs Training-Vs-Feature Image Conv2d Eval"], marker='o', label="SSIM Train_Input-Conv")
plt.plot(df["SSIM/DTW"], df["SSIM Inputs Training-Vs-Feature Image FC Eval"], marker='o', label="SSIM Train_Input-FC")
plt.plot(df["SSIM/DTW"], df["Evaluation Threshold Accuracy 1 dp (%)"] / 100, marker='o', label="Eval Threshold 1 dp")
plt.plot(df["SSIM/DTW"], df["Evaluation MAE"], marker='o', label="Evaluation MAE")
plt.plot(df["SSIM/DTW"], df["Evaluation R^2"], marker='o', label="Evaluation R^2")

# Adding labels and title
plt.xlabel("SSIM/DTW")
plt.ylabel("Metrics")
plt.title("Comparison of Metrics vs SSIM/DTW")
plt.legend()
plt.grid(visible=True, linestyle='--', alpha=0.5)

# Modify the x_labels to exclude "PWBK"
x_labels = [f"{name}\n{round(value, 4)}" if name != "PWBK" else "" for name, value in zip(df["Eval Stock"], df["SSIM/DTW"])]
plt.xticks(df["SSIM/DTW"], x_labels, rotation=45, ha='right', rotation_mode='anchor')

# Move the "PWBK" label slightly higher using plt.text
pwbk_x = df[df["Eval Stock"] == "PWBK"]["SSIM/DTW"].values[0]
pwbk_y = -0.5  # Adjust this value as needed to move the label higher
plt.text(pwbk_x, pwbk_y, "PWBK", ha='center', va='bottom', fontsize=10, color='black')

plt.tight_layout()
plt.show()
