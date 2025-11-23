import numpy as np
from iou import IOU


def calculate_pairs(boxes, true_boxes, IOU_threshold):
    box_pairs = []
    for i, box in enumerate(boxes):
        for j, true_box in enumerate(true_boxes):
            if IOU(box, true_box) > IOU_threshold:
                box_pairs.append([i,j])
    return np.array(box_pairs)

def compute_metrics(boxes, scores, true_boxes, threshold, box_pairs):
    TP = 0
    FP = 0
    FN = 0
    
    checked_boxes = np.full(len(boxes), False)
    checked_true_boxes = np.full(len(true_boxes), False)

    for i in range(len(boxes)):
        if i in box_pairs[:,0]:
            if scores[i] > threshold:
                TP += 1
            else:
                FN += 1
            checked_true_boxes[box_pairs[i,1]] = True
        else:
            FP += 1
        checked_boxes[i] = True
    
    FN += np.sum(~checked_true_boxes)

    recall = TP / (TP + FN)
    precision = TP / (TP + FP)

    return (recall, precision)

def average_precision(boxes, scores, true_boxes, IOU_threshold):
    thresholds = np.linspace(0, 1, 21)
    box_pairs = calculate_pairs(boxes, true_boxes, IOU_threshold)
    RP = np.zeros((len(thresholds), 2))
    AP = 0

    for i, threshold in enumerate(thresholds):
        recall, precision = compute_metrics(boxes, scores, true_boxes, threshold, box_pairs)
        RP[i,0] = recall
        RP[i,1] = precision
        if i > 0:
            AP += (RP[i,0] - RP[i-1,0]) * (RP[i,1] + RP[i-1,1])/2
    
    return AP, RP