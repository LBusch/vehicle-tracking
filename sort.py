import numpy as np
from detection_assignment import assign_detections_to_trackers
from filter import SortKalmanFilter

def bbox_to_measurement(bbox):
    """convert bounding box to measurement vector for Kalman filter."""

    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    if width <= 0 or height <= 0:
        raise ValueError("Bounding box must have positive width and height.")

    cx = x1 + width / 2.0
    cy = y1 + height / 2.0
    scale = width * height
    aspect_ratio = width / height if height > 0 else 1.0

    return np.array([cx, cy, scale, aspect_ratio], dtype=np.float32)

def measurement_to_bbox(measurement):
    """convert measurement vector back to bounding box."""

    cx, cy, scale, aspect_ratio = measurement
    if scale <= 0 or aspect_ratio <= 0:
        raise ValueError("Scale and aspect ratio must be positive.")
    width = np.sqrt(scale * aspect_ratio)
    height = scale / width
    x1 = cx - width / 2.0
    y1 = cy - height / 2.0
    x2 = cx + width / 2.0
    y2 = cy + height / 2.0

    return np.array([x1, y1, x2, y2], dtype=np.float32)

class KalmanBoxTracker:
    """This class represents the state of individual tracked objects with a Kalman filter and tracking information."""
    count = 0

    def __init__(self, bbox):
        # create a Kalman filter for this tracker
        self.kf = SortKalmanFilter(bbox_to_measurement(bbox))
        # assign a unique ID to the tracker
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.time_since_update = 0  # number of frames since last detection
        self.hit_streak = 0         # consecutive matched detections
        self.age = 0                # number of frames since the tracker was created

    def update(self, bbox):
        """Update the state vector with matched detection."""
        self.time_since_update = 0
        self.hit_streak += 1
        measurement = bbox_to_measurement(bbox)
        self.kf.update(measurement)

    def predict(self):
        """Advance the state vector and return the predicted bounding box estimate."""
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.kf.predict()

        predicted_bbox = measurement_to_bbox(self.kf.get_measurement())
        return predicted_bbox

    def get_bbox(self):
        """Return the current bounding box estimate."""
        return measurement_to_bbox(self.kf.get_measurement())

class Sort:
    """This class implements the SORT tracker."""

    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age              # maximum number of frames to keep a tracker without assigned detections
        self.min_hits = min_hits            # minimum number of assigned detections before a tracker is considered valid
        self.iou_threshold = iou_threshold  # minimum IoU for matching detections to trackers

        self.trackers = []
        self.frame_count = 0

    def update(self, detections):
        """
        Update the state of the tracker with new detections.

        detections: a numpy array of shape (N, 4) where N is the number of detections and each detection is [x1, y1, x2, y2].
        Returns a list of tracked objects in the format [x1, y1, x2, y2, id].
        """

        self.frame_count += 1

        # Get predicted locations from existing trackers
        predicted_boxes = np.array([tracker.predict() for tracker in self.trackers])

        # Assign detections to trackers
        matches, unmatched_detections, unmatched_trackers = assign_detections_to_trackers(
            detections, predicted_boxes, self.iou_threshold
        )

        # Update matched trackers with assigned detections
        for tracker_idx, detection_idx in matches:
            self.trackers[tracker_idx].update(detections[detection_idx])

        # Create new trackers for unmatched detections
        for detection_idx in unmatched_detections:
            new_tracker = KalmanBoxTracker(detections[detection_idx])
            self.trackers.append(new_tracker)

        # Remove dead trackers (those that have not been updated for max_age frames)
        self.trackers = [
            tracker for tracker in self.trackers if tracker.time_since_update <= self.max_age
        ]

        # Only return trackers that have been hit at least min_hits times
        output_tracks = []
        for tracker in self.trackers:
            if (tracker.hit_streak >= self.min_hits) or (self.frame_count <= self.min_hits):
                output_tracks.append(np.concatenate((tracker.get_bbox(), [tracker.id + 1])))

        return np.array(output_tracks) if len(output_tracks) > 0 else np.empty((0, 5))