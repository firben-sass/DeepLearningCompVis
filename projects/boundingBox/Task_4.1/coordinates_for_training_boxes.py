import cv2
import os
import numpy as np
import csv
import xml.etree.ElementTree as ET

images_folder    = "/Users/dani/Desktop/IDL_project4/projects/boundingBox/potholes/images"
xml_folder       = "/Users/dani/Desktop/IDL_project4/projects/boundingBox/potholes/annotations"
proposals_folder = "/Users/dani/Desktop/IDL_project4/projects/boundingBox/proposals"

RESIZE_W, RESIZE_H = 600, 600
K = 1000  #Based on recall experiments

POS_IOU_THRESH = 0.5
NEG_IOU_THRESH = 0.3


def xywh_to_xyxy(box):
    x, y, w, h = box
    return [x, y, x + w, y + h]

def iou(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    xA = max(x1_min, x2_min)
    yA = max(y1_min, y2_min)
    xB = min(x1_max, x2_max)
    yB = min(y1_max, y2_max)

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h

    area1 = max(0, x1_max - x1_min) * max(0, y1_max - y1_min)
    area2 = max(0, x2_max - x2_min) * max(0, y2_max - y2_min)

    union = area1 + area2 - inter_area + 1e-6
    return inter_area / union

def parse_gt_boxes_resized(xml_path, w_orig, h_orig, resize_w, resize_h):
    scale_x = resize_w / float(w_orig)
    scale_y = resize_h / float(h_orig)

    tree = ET.parse(xml_path)
    root = tree.getroot()

    gt_boxes = []
    for obj in root.findall("object"):
        bnd = obj.find("bndbox")
        xmin = int(float(bnd.find("xmin").text))
        ymin = int(float(bnd.find("ymin").text))
        xmax = int(float(bnd.find("xmax").text))
        ymax = int(float(bnd.find("ymax").text))

        xmin_r = int(round(xmin * scale_x))
        xmax_r = int(round(xmax * scale_x))
        ymin_r = int(round(ymin * scale_y))
        ymax_r = int(round(ymax * scale_y))

        gt_boxes.append([xmin_r, ymin_r, xmax_r, ymax_r])
    return gt_boxes



train_samples = []  # rows: [image, x, y, w, h, label]

image_files = [f for f in os.listdir(images_folder)
               if f.lower().endswith((".jpg", ".png", ".jpeg"))]

for img_name in image_files:
    img_path = os.path.join(images_folder, img_name)
    xml_path = os.path.join(xml_folder, os.path.splitext(img_name)[0] + ".xml")
    prop_path = os.path.join(proposals_folder, img_name + ".npy")

    if not os.path.exists(xml_path) or not os.path.exists(prop_path):
        continue

    img = cv2.imread(img_path)
    if img is None:
        continue
    h_orig, w_orig = img.shape[:2]

    gt_boxes = parse_gt_boxes_resized(xml_path, w_orig, h_orig, RESIZE_W, RESIZE_H)
    if len(gt_boxes) == 0:
        continue

    # load TOP K proposals
    boxes = np.load(prop_path)[:K]
    proposals_xyxy = [xywh_to_xyxy(b) for b in boxes]

    for b_xywh, b_xyxy in zip(boxes, proposals_xyxy):
        if len(gt_boxes) == 0:
            max_iou = 0.0
        else:
            max_iou = max(iou(b_xyxy, gt) for gt in gt_boxes)

        if max_iou >= POS_IOU_THRESH:
            label = 1
        elif max_iou <= NEG_IOU_THRESH:
            label = 0
        else:
            continue  # skip ambiguous

        x, y, w, h = b_xywh
        train_samples.append([img_name, x, y, w, h, label])



# Save as numpy
train_samples_np = np.array(train_samples, dtype=object)
np.save(os.path.join(proposals_folder, "train_samples.npy"), train_samples_np)
print("Saved:", os.path.join(proposals_folder, "train_samples.npy"))

# # Save as CSV
# csv_path = os.path.join(proposals_folder, "train_samples.csv")
# with open(csv_path, "w", newline="") as f:
#     writer = csv.writer(f)
#     writer.writerow(["image", "x", "y", "w", "h", "label"])
#     writer.writerows(train_samples)

#print("Saved:", csv_path)
print("Total labeled proposals:", len(train_samples))
