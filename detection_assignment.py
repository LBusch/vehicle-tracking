import numpy as np
from scipy.optimize import linear_sum_assignment

def iou(bbox1, bbox2):
    """compute intersection over union of two bounding boxes."""

    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    union_area = bbox1_area + bbox2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0.0

def build_iou_matrix(tracker_boxes, detections):
    """build IoU matrix between detections and tracker boxes."""

    iou_matrix = np.zeros((len(tracker_boxes), len(detections)), dtype=np.float32)
    for i, tracker in enumerate(tracker_boxes):
        for j, detection in enumerate(detections):
            iou_matrix[i, j] = iou(tracker, detection)
    return iou_matrix

def assign_detections_to_trackers(detections, tracker_boxes, iou_threshold=0.3):
    """
    Assign detections to trackers using the Hungarian algorithm.
    Returns indeces of matched detections and trackers, as well as unmatched detections and trackers.
    """

    num_trackers = len(tracker_boxes)
    num_detections = len(detections)

    # Handle cases where one or both inputs are empty
    if num_trackers == 0 or num_detections == 0:
        return (
            np.empty((0, 2), dtype=int),
            list(range(num_detections)),
            list(range(num_trackers)),
        )

    iou_matrix = build_iou_matrix(tracker_boxes, detections)

    # Find the assignment with the highest total IoU (minimize the cost of -IoU)
    row_indices, col_indices = linear_sum_assignment(-iou_matrix)

    matches = []
    unmatched_trackers = set(range(num_trackers))
    unmatched_detections = set(range(num_detections))

    for tracker_idx, detection_idx in zip(row_indices, col_indices):
        # Reject assignments with insufficient overlap
        if iou_matrix[tracker_idx, detection_idx] < iou_threshold:
            continue

        matches.append([tracker_idx, detection_idx])
        unmatched_trackers.remove(tracker_idx)
        unmatched_detections.remove(detection_idx)

    return np.array(matches), list(unmatched_detections), list(unmatched_trackers)