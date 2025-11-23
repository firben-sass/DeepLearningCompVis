import numpy as np
from iou import IOU


def nms(boxes, confidence_scores, confidence_threshold=0.7, IOU_threshold=0.4):
    print(np.sort(confidence_scores))
    boxes = boxes[np.argsort(confidence_scores)]
    boxes_filtered = boxes[np.where(confidence_scores > confidence_threshold)]
    boxes_removed = np.full(len(boxes_filtered), False)
    for i, box in enumerate(boxes_filtered):
        for j in range(i+1, len(boxes_filtered)):
            if boxes_removed[j]:
                continue
            iou = IOU(box, boxes_filtered[j])
            if iou > IOU_threshold:
                boxes_removed[j] = True
    boxes_filtered = boxes_filtered[~boxes_removed]
    return boxes_filtered