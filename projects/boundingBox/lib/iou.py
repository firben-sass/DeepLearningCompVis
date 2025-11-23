import numpy as np


def IOU(box1, box2):
    x1, y1, x2, y2 = box1	
    x3, y3, x4, y4 = box2

    width_box1 = abs(x2 - x1)
    height_box1 = abs(y2 - y1)
    width_box2 = abs(x4 - x3)
    height_box2 = abs(y4 - y3)
    area_box1 = width_box1 * height_box1
    area_box2 = width_box2 * height_box2

    x_overlap1 = max(x1, x3)
    y_overlap1 = max(y1, y3)
    x_overlap2 = min(x2, x4)
    y_overlap2 = min(y2, y4)
    width_overlap = abs(x_overlap2 - x_overlap1)
    height_overlap = abs(y_overlap2 - y_overlap1)

    area_overlap = width_overlap * height_overlap
    area_union = area_box1 + area_box2 - area_overlap
    IOU = area_overlap / area_union
    return IOU