import xml.etree.ElementTree as ET
import cv2


def xywh_to_xyxy(b):
    x, y, w, h = b
    return [x, y, x + w, y + h]

def parse_gt_boxes_resized(xml_path, resize_w, resize_h):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    w_orig = float(root.find("size/width").text)
    h_orig = float(root.find("size/height").text)

    scale_x = resize_w / w_orig
    scale_y = resize_h / h_orig

    boxes = []
    for obj in root.findall("object"):
        b = obj.find("bndbox")
        xmin = int(float(b.find("xmin").text) * scale_x)
        ymin = int(float(b.find("ymin").text) * scale_y)
        xmax = int(float(b.find("xmax").text) * scale_x)
        ymax = int(float(b.find("ymax").text) * scale_y)
        boxes.append([xmin, ymin, xmax, ymax])
    return boxes