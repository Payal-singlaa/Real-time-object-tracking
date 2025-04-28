from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
import time

model = YOLO('yolov8s.pt')
tracker = DeepSort(max_age=30, n_init=3)

frame_count = 0
fps_sum = 0

cap = cv2.VideoCapture(0)

current_ids = set()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    start_time = time.time()

    results = model.predict(frame, imgsz=320, conf=0.5)[0]

    det_list = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        conf = box.conf[0]
        cls = box.cls[0]
        det_list.append(([x1.item(), y1.item(), (x2-x1).item(), (y2-y1).item()], conf.item(), int(cls)))

    tracks = tracker.update_tracks(det_list, frame=frame)

    frame_ids = set()

    new_object_detected = False
    missing_object_detected = False

    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        l, t, w, h = track.to_ltrb()
        frame_ids.add(track_id)
        cv2.rectangle(frame, (int(l), int(t)), (int(w), int(h)), (0, 255, 0), 2)
        cv2.putText(frame, f'ID: {track_id}', (int(l), int(t) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    missing_ids = current_ids - frame_ids
    new_ids = frame_ids - current_ids

    for mid in missing_ids:
        print(f"Object ID {mid} is missing.")
        missing_object_detected = True

    for nid in new_ids:
        print(f"New Object ID {nid} detected.")
        new_object_detected = True

    current_ids = frame_ids

    if new_object_detected:
        cv2.putText(frame, "New Object Detected!", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    if missing_object_detected:
        cv2.putText(frame, "Object Missing!", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    end_time = time.time()
    fps = 1 / (end_time - start_time)

    frame_count += 1
    fps_sum += fps

    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    cv2.imshow('Real-Time Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

average_fps = fps_sum / frame_count
print(f"Average FPS: {average_fps:.2f}")