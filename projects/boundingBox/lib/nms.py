import numpy as np
from boundingBox.lib.tools.iou import IOU


def nms(boxes, confidence_scores, confidence_threshold=0.7, IOU_threshold=0.5):
    order = np.argsort(confidence_scores)
    confidence_scores = confidence_scores[order]
    boxes = boxes[order]
    confidence_scores_filtered = confidence_scores[confidence_scores > confidence_threshold]
    boxes_filtered = boxes[confidence_scores > confidence_threshold]
    boxes_removed = np.full(len(boxes_filtered), False)
    for i, box in enumerate(boxes_filtered):
        for j in range(i+1, len(boxes_filtered)):
            if boxes_removed[j]:
                continue
            iou = IOU(box, boxes_filtered[j])
            if iou > IOU_threshold:
                boxes_removed[j] = True
    confidence_scores_filtered = confidence_scores_filtered[~boxes_removed]
    boxes_filtered = boxes_filtered[~boxes_removed]
    return boxes_filtered, confidence_scores_filtered