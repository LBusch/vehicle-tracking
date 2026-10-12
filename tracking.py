import cv2
import os
from ultralytics import YOLO
from sort import Sort

img_sequence_dir = "data/DETRAC-Images/MVI_40981" # 40981
video_name = "MVI_40981.mp4"
trained_model_path = "runs/detect/yolo26s_15epochs/weights/best.pt"
use_tracker = True  # Set to True to enable tracking, False for detection only


def draw_tracked_bboxes(frame, tracked_boxes):
    for i, box in enumerate(tracked_boxes):
        x1, y1, x2, y2, id = box
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
        cv2.putText(frame, f"ID {int(id)}", (x1, max(y1 - 10, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
def draw_detections(frame, detections):
    for det in detections:
        x1, y1, x2, y2 = map(int, det[:4])
        cv2.rectangle(frame, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)


model = YOLO(trained_model_path)
tracker = Sort(max_age=1, min_hits=3, iou_threshold=0.3)
images = sorted([img for img in os.listdir(img_sequence_dir) if img.endswith('.jpg')])
frame = cv2.imread(os.path.join(img_sequence_dir, images[0]))
height, width, _ = frame.shape
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(video_name, fourcc, 30.0, (width, height))

for img_name in images:
    img_path = os.path.join(img_sequence_dir, img_name)
    frame = cv2.imread(img_path)
    results = model(frame, verbose=False)

    if use_tracker:
        # Use SORT for tracking
        detections = results[0].boxes.xyxy.cpu().numpy().astype(float)
        tracked_boxes = tracker.update(detections)
        draw_tracked_bboxes(frame, tracked_boxes)
    else:
        detections = results[0].boxes.xyxy.cpu().numpy().astype(float)
        draw_detections(frame, detections)
    
    draw_tracked_bboxes(frame, tracked_boxes)
    video_writer.write(frame)
    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_writer.release()
cv2.destroyAllWindows()