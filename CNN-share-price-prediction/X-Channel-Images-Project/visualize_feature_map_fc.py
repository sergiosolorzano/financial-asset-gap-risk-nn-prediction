import numpy as np
import matplotlib.pyplot as plt

# Load the .npz file
feature_maps = np.load('models/model_checkpoints/feature_maps_fc_CFG_KEY_stable_91.npz')

# List all arrays stored in the file
print("Keys in the .npz file:", feature_maps.files)

# Access one specific feature map
key = feature_maps.files[0]  # Use the first key as an example
feature_map = feature_maps[key]

print(f"Shape of the feature map ({key}):", feature_map.shape)

# Check dimensions and visualize accordingly
if len(feature_map.shape) == 2:  # 2D array, likely (samples, features)
    # Visualize all samples for the first feature
    plt.plot(feature_map[:, 0], marker='o')
    plt.title(f"Feature Map (Key: {key})")
    plt.xlabel("Sample Index")
    plt.ylabel("Feature Value")
    plt.grid()
    plt.show()
else:
    print("Unexpected feature map shape, update visualization logic as needed.")
