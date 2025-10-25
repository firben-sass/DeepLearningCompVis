import matplotlib.pyplot as plt
import torch

# Create a small dataset instance (adjust the path as needed)
dataset = OpticalFlowDataset(root_dir="/path/to/ucf101_noleakage", split='train', image_size=224)
print(f"Total videos: {len(dataset)}")
print(f"Classes: {dataset.classes}")

# Get one example
flows, label = dataset[0]
print(f"Flow tensor shape: {flows.shape}")  # should be [18, 224, 224]
print(f"Label index: {label} ({dataset.classes[label]})")

# --- Visualize the first few flow components ---
# Remember: channels [0,1] = flow_1 (dx, dy), [2,3] = flow_2, etc.
num_pairs = flows.shape[0] // 2
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
for i in range(4):
    dx = flows[2*i].numpy()
    dy = flows[2*i + 1].numpy()

    axes[0, i].imshow(dx, cmap='jet')
    axes[0, i].set_title(f'flow_{i+1}: dx (horizontal)')
    axes[1, i].imshow(dy, cmap='jet')
    axes[1, i].set_title(f'flow_{i+1}: dy (vertical)')

for ax in axes.ravel():
    ax.axis('off')

plt.tight_layout()
plt.show()
