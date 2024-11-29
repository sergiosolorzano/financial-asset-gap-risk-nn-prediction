import numpy as np

# Load the .npz file
#feature_maps = np.load('models/model_checkpoints/feature_maps_cnn_CFG_KEY_stable_91.npz')
feature_maps = np.load('models/model_checkpoints/feature_maps_cnn_CFG_KEY_stable_91.npz')

# List all arrays stored in the file
print("Keys in the .npz file:", feature_maps.files)

# Access one specific feature map
key = feature_maps.files[0]  # Use the first key as an example
feature_map = feature_maps[key]

print(f"Shape of the feature map ({key}):", feature_map.shape)

# Access one specific feature map
key = feature_maps.files[0]  # Use the first key as an example
feature_map = feature_maps[key]

print(f"Shape of the feature map ({key}):", feature_map.shape)

import matplotlib.pyplot as plt

# Select one channel (assuming 4D: [batch_size, channels, height, width])
channel_idx = 0
plt.imshow(feature_map[0, channel_idx, :, :], cmap='viridis')  # Display the first sample and first channel
plt.title(f"Feature Map (Key: {key}, Channel: {channel_idx})")
plt.colorbar()
plt.show()

def plot_feature_maps(feature_map, num_channels=6):
    """
    Plot a grid of feature maps from the first sample.
    """
    plt.figure(figsize=(15, 10))
    for i in range(min(num_channels, feature_map.shape[1])):
        plt.subplot(1, num_channels, i + 1)
        plt.imshow(feature_map[0, i, :, :], cmap='viridis')
        plt.title(f"Channel {i}")
        plt.axis('off')
    plt.show()

# Plot the first 6 channels
plot_feature_maps(feature_map, num_channels=6)
