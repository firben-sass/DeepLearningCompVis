import matplotlib.pyplot as plt
import torch
from datasets import OpticalFlowDataset

# Create a small dataset instance (adjust the path as needed)
dataset = OpticalFlowDataset(root_dir="/home/datawa/Repositories/ucf101_noleakage", split='train', image_size=224)
flows, label = dataset[0]

print(f"Flow tensor shape: {flows.shape}")  # expect [18, 224, 224]
print(f"Label: {label} ({dataset.classes[label]})")

# --- Basic stats ---
print("\nPer-channel statistics:")
for i in range(flows.shape[0]):
    ch = flows[i]
    print(f"  Channel {i:02d}: min={ch.min():.2f}, max={ch.max():.2f}, mean={ch.mean():.2f}")

# --- Global histogram ---
flat = flows.flatten().numpy()
plt.figure(figsize=(6,4))
plt.hist(flat, bins=100, color='skyblue', edgecolor='k')
plt.title("Distribution of flow values across all channels")
plt.xlabel("Flow value")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

# --- Visualize a few flow fields (dx, dy) ---
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