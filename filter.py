import numpy as np
from filterpy.kalman import KalmanFilter

class SortKalmanFilter:
    """A Kalman filter for tracking bounding boxes in SORT."""
    def __init__(self, init_state):
        """
        State:
        [cx, cy, scale, aspect_ratio, vx, vy, vscale]

        Measurement:
        [cx, cy, scale, aspect_ratio]
        """
        self.kf = KalmanFilter(dim_x=7, dim_z=4)

        # initial state vector
        self.kf.x = np.zeros(7)
        self.kf.x[:4] = init_state

        # state transition matrix (assuming constant velocity model)
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
        ], dtype=np.float32)

        # measurement matrix
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0 ,0],
        ], dtype=np.float32)

        # process noise covariance
        self.kf.Q = np.eye(7)
        self.kf.Q[-1, -1] *= 0.01  # low process noise for scale changes
        self.kf.Q[4:, 4:] *= 0.01  # low process noise for velocities for smoother tracking

        # measurement noise covariance
        self.kf.R = np.eye(4)
        self.kf.R[2:, 2:] *= 10.0  # higher measurement noise for scale and aspect ratio

        # state uncertainty covariance
        self.kf.P = np.eye(7)
        self.kf.P[4:, 4:] *= 1000.0  # High uncertainty for velocities
        self.kf.P *= 10.0  # Overall uncertainty scaling

    def predict(self):
        # avoid predicting negative scale by setting scale velocity to zero
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] = 0.0  
        self.kf.predict()
        return self.kf.x.copy()

    def update(self, measurement):
        self.kf.update(measurement)
        return self.kf.x.copy()

    def get_measurement(self):
        return self.kf.x[:4].copy()