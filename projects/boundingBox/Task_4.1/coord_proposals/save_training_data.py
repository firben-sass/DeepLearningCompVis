import cv2
import os
import numpy as np


#its Chatgpt

# ========= CONFIG (CHANGE HERE) =========
images_folder    = "/Users/dani/Desktop/IDL_project4/projects/boundingBox/potholes/images"
proposals_folder = "/Users/dani/Desktop/IDL_project4/projects/boundingBox/proposals"
PROPOSAL_FILE    = os.path.join(proposals_folder, "reduced_coords_xyxy.npy")

RESIZE_W, RESIZE_H = 600, 600
CROP_SIZE = 224

save_root = os.path.join(proposals_folder, "dataset_crops")
save_positive_dir = os.path.join(save_root, "potholes")
save_negative_dir = os.path.join(save_root, "background")

os.makedirs(save_positive_dir, exist_ok=True)
os.makedirs(save_negative_dir, exist_ok=True)

# ========= LOAD REDUCED COORDS =========
data = np.load(PROPOSAL_FILE, allow_pickle=True)
print("Loaded reduced dataset:", data.shape)

count_pos = 0
count_neg = 0

# ========= PROCESS EACH SAMPLE =========
for row in data:
    img_name, x1, y1, x2, y2, label = row
    img_name = str(img_name)

    # Load image
    img_path = os.path.join(images_folder, img_name)
    img = cv2.imread(img_path)
    if img is None:
        print(f"[WARN] Could not read {img_path}")
        continue

    # Resize to match proposal coordinate scale
    img_resized = cv2.resize(img, (RESIZE_W, RESIZE_H))

    # Crop
    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
    crop = img_resized[y1:y2, x1:x2]

    # Skip invalid/empty crops
    if crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 5:
        continue

    # Resize crop to target size (easily changeable)
    crop_resized = cv2.resize(crop, (CROP_SIZE, CROP_SIZE))

    # Save according to label
    if label == 1:
        out_name = f"pothole_{count_pos}.png"
        cv2.imwrite(os.path.join(save_positive_dir, out_name), crop_resized)
        count_pos += 1
    else:
        out_name = f"background_{count_neg}.png"
        cv2.imwrite(os.path.join(save_negative_dir, out_name), crop_resized)
        count_neg += 1

# ========= STATS =========
print("Saved cropped dataset to:", save_root)
print(f"Potholes saved:   {count_pos}")
print(f"Background saved: {count_neg}")
print(f"Crop size: {CROP_SIZE}x{CROP_SIZE}")
