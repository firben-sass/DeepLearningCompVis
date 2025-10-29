import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from glob import glob
from PIL import Image
from datasets import OpticalFlowDataset
import random

# --- Load dataset ---
dataset = OpticalFlowDataset(
    root_dir="/home/datawa/Repositories/ucf101_noleakage",
    split="train",
    image_size=224
)

# --- Select two random samples from different classes ---
indices = random.sample(range(len(dataset)), 2)

fig, axes = plt.subplots(2, 3, figsize=(10, 7))

for row, idx in enumerate(indices):
    flows, label = dataset[idx]
    class_name = dataset.classes[label]
    video_path = dataset.video_paths[idx]

    # --- Load corresponding RGB frame ---
    rgb_dir = video_path.replace("flows", "frames")
    rgb_files = sorted(glob(os.path.join(rgb_dir, "*.jpg")) + glob(os.path.join(rgb_dir, "*.png")))
    if len(rgb_files) == 0:
        raise FileNotFoundError(f"No RGB frames found in {rgb_dir}")
    rgb_img = np.array(Image.open(rgb_files[0]).convert("RGB"))

    # --- Extract first optical flow pair ---
    dx = flows[0].numpy()
    dy = flows[1].numpy()

    # --- Plot RGB + flow maps ---
    axes[row, 0].imshow(rgb_img)
    axes[row, 0].set_title(f"Class: {class_name}")
    axes[row, 0].axis("off")

    axes[row, 1].imshow(dx, cmap="jet")
    axes[row, 1].set_title("Horizontal flow(dx)")
    axes[row, 1].axis("off")

    axes[row, 2].imshow(dy, cmap="jet")
    axes[row, 2].set_title("Vertical flow (dy)")
    axes[row, 2].axis("off")

plt.tight_layout()
plt.savefig("plots/flow_examples.png", dpi=300, bbox_inches="tight")
plt.show()
