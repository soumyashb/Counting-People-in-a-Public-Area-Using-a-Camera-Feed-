import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort


VIDEO_PATH = "test 2.mp4"
GROUND_TRUTH_COUNT = 28       
CONF_THRESHOLD = 0.5

model = YOLO("yolov8n.pt")

tracker = DeepSort(
    max_age=60,              
    n_init=3,
    max_iou_distance=0.7
)

cap = cv2.VideoCapture(VIDEO_PATH)
unique_ids = set()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]
    detections = []

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])

        if cls_id == 0 and conf >= CONF_THRESHOLD:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w = x2 - x1
            h = y2 - y1
            detections.append(([x1, y1, w, h], conf, "person"))

    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        unique_ids.add(track_id)

        l, t, r, b = map(int, track.to_ltrb())

        cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)
        cv2.putText(frame,
                    f"ID {track_id}",
                    (l, t - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2)

    predicted_count = len(unique_ids)
    error = abs(predicted_count - GROUND_TRUTH_COUNT)

    if GROUND_TRUTH_COUNT > 0:
        accuracy = (1 - (error / GROUND_TRUTH_COUNT)) * 100
    else:
        accuracy = 0

    cv2.putText(frame,
                f"Total Count: {predicted_count}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2)

    cv2.putText(frame,
                f"Live Accuracy: {accuracy:.2f}%",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2)

    cv2.imshow("Real-Time Tracking & Accuracy", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()