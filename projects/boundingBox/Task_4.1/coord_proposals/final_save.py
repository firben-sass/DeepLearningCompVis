import cv2
import os
import numpy as np
from collections import defaultdict

# ========= CONFIG =========
images_folder = "/work3/s204164/DeepLearningCompVis/projects/boundingBox/data/images"
proposals_folder = "/work3/s204164/DeepLearningCompVis/projects/boundingBox/Task_4.1/coord_proposals"
PROPOSAL_FILE = os.path.join(proposals_folder, "reduced_coords_xyxy.npy")
RESIZE_W, RESIZE_H = 600, 600
CROP_SIZE = 224

save_root = os.path.join(proposals_folder, "dataset_crops_split")
splits = ['train', 'val', 'test']

# Create directory structure
for split in splits:
    os.makedirs(os.path.join(save_root, split, "potholes"), exist_ok=True)
    os.makedirs(os.path.join(save_root, split, "background"), exist_ok=True)

# ========= LOAD AND GROUP BY IMAGE =========
data = np.load(PROPOSAL_FILE, allow_pickle=True)
print("Loaded reduced dataset:", data.shape)

# Group proposals by image name
image_proposals = defaultdict(list)
for row in data:
    img_name = str(row[0])
    image_proposals[img_name].append(row)

# Get unique images and shuffle
unique_images = list(image_proposals.keys())
np.random.seed(42)  # For reproducibility
np.random.shuffle(unique_images)

total_images = len(unique_images)
print(f"Total unique images: {total_images}")

# Split images: 70% train, 15% val, 15% test
train_split = int(0.70 * total_images)
val_split = int(0.85 * total_images)

train_images = unique_images[:train_split]
val_images = unique_images[train_split:val_split]
test_images = unique_images[val_split:]

print(f"Train images: {len(train_images)}")
print(f"Val images: {len(val_images)}")
print(f"Test images: {len(test_images)}")

# ========= PROCESS AND SAVE CROPS =========
def process_split(image_list, split_name):
    count_pos = 0
    count_neg = 0
    
    for img_name in image_list:
        proposals = image_proposals[img_name]
        
        # Load image once for all proposals
        img_path = os.path.join(images_folder, img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARN] Could not read {img_path}")
            continue
        
        # Resize to match proposal coordinate scale
        img_resized = cv2.resize(img, (RESIZE_W, RESIZE_H))
        
        # Process all proposals for this image
        for row in proposals:
            _, x1, y1, x2, y2, label = row
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            
            # Crop
            crop = img_resized[y1:y2, x1:x2]
            
            # Skip invalid/empty crops
            if crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 5:
                continue
            
            # Resize crop to target size
            crop_resized = cv2.resize(crop, (CROP_SIZE, CROP_SIZE))
            
            # Save according to label
            if label == 1:
                out_name = f"pothole_{split_name}_{count_pos}.png"
                save_path = os.path.join(save_root, split_name, "potholes", out_name)
                cv2.imwrite(save_path, crop_resized)
                count_pos += 1
            else:
                out_name = f"background_{split_name}_{count_neg}.png"
                save_path = os.path.join(save_root, split_name, "background", out_name)
                cv2.imwrite(save_path, crop_resized)
                count_neg += 1
    
    return count_pos, count_neg

# Process each split
print("\n========= Processing Train Split =========")
train_pos, train_neg = process_split(train_images, 'train')
print(f"Train - Potholes: {train_pos}, Background: {train_neg}")

print("\n========= Processing Val Split =========")
val_pos, val_neg = process_split(val_images, 'val')
print(f"Val - Potholes: {val_pos}, Background: {val_neg}")

print("\n========= Processing Test Split =========")
test_pos, test_neg = process_split(test_images, 'test')
print(f"Test - Potholes: {test_pos}, Background: {test_neg}")

print(f"\n========= COMPLETE =========")
print(f"Dataset saved to: {save_root}")
print(f"Crop size: {CROP_SIZE}x{CROP_SIZE}")

# Save split information for reference
split_info = {
    'train_images': train_images,
    'val_images': val_images,
    'test_images': test_images
}
np.save(os.path.join(save_root, 'split_info.npy'), split_info)