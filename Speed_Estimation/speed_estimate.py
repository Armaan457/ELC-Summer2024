import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from collections import deque

model = YOLO("yolov8n.pt")

video_path = 'data/vehicles.mp4'
output_path = 'data/answer.mp4'
speed_limit = 80 

cap = cv2.VideoCapture(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'X264')
out = cv2.VideoWriter(output_path, fourcc, frame_rate, (width, height))

line_pts = [(0, int(height * 0.5)+120), (width, int(height * 0.5)+120)]

tracker = DeepSort(max_age=30, nn_budget=100)

trackers = {}

emoji = cv2.imread('assets/angry_emoji.png', cv2.IMREAD_UNCHANGED)
emoji2 = cv2.imread('assets/happy_emoji.png', cv2.IMREAD_UNCHANGED)

if emoji.shape[2] == 3:  # If the image doesn't have an alpha channel, add one
    emoji = cv2.cvtColor(emoji, cv2.COLOR_BGR2BGRA)
if emoji2.shape[2] == 3:  # If the image doesn't have an alpha channel, add one
    emoji2 = cv2.cvtColor(emoji2, cv2.COLOR_BGR2BGRA)

def calculate_speed(pixel_distance, fps, ppm=12):
    distance_meters = pixel_distance / ppm
    speed_mps = distance_meters * fps 
    speed_kmph = speed_mps * 3.6  # convert to km/h
    return speed_kmph

def overlay_image_alpha(img, img_overlay, x, y, alpha_mask):
    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])


    y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
    x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return

    img_crop = img[y1:y2, x1:x2]
    img_overlay_crop = img_overlay[y1o:y2o, x1o:x2o]
    alpha = alpha_mask[y1o:y2o, x1o:x2o, np.newaxis]

    img_crop[:] = alpha * img_overlay_crop + (1 - alpha) * img_crop

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]

    detections = []
    for result in results.boxes:
        if result.cls.item() in [2, 5, 7] and result.conf.item() > 0.5: 
            x1, y1, x2, y2 = map(int, result.xyxy[0])
            bbox = [x1, y1, x2-x1, y2-y1]
            score = result.conf.item()
            detections.append((bbox, score))

    tracks = tracker.update_tracks(detections, frame=frame)


    for track in tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue

        track_id = track.track_id
        l, t, r, b = track.to_ltrb()  


        center_x = (l+r) / 2
        center_y = (t+b) / 2

        if track_id not in trackers:
            trackers[track_id] = {'positions': deque(maxlen=2), 'crossed_line': False, 'speed': 0}

        trackers[track_id]['positions'].append((center_x, center_y))

        # Check if vehicle has crossed the line
        if len(trackers[track_id]['positions']) == 2:
            (x0, y0) = trackers[track_id]['positions'][0]
            (x1, y1) = trackers[track_id]['positions'][1]

            m = (line_pts[1][1] - line_pts[0][1]) / (line_pts[1][0] - line_pts[0][0] + 1e-6)
            c = line_pts[0][1] - m * line_pts[0][0]

            if (y0 < m * x0 + c and y1 >= m * x1 + c) or (y0 >= m * x0 + c and y1 < m * x1 + c):
                distance = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
                speed = calculate_speed(distance, frame_rate)
                trackers[track_id]['speed'] = speed
                trackers[track_id]['crossed_line'] = True


        cv2.rectangle(frame, (int(l), int(t)), (int(r), int(b)), (0, 255, 0), 2)
        speed_text = f"{trackers[track_id]['speed']:.2f} km/h"
        cv2.putText(frame, speed_text, (int(l)+20, int(t)+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


        if trackers[track_id]['crossed_line'] and trackers[track_id]['speed'] > speed_limit:
            # warning_text = "OVERSPEEDING"
            # cv2.putText(frame, warning_text, (int(l), int(b) + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            emoji_resized = cv2.resize(emoji, (80, 80))
            overlay_image_alpha(frame, emoji_resized[:, :, :3], int(l), int(b) + 40, emoji_resized[:, :, 3] / 255.0)
        elif trackers[track_id]['crossed_line'] and trackers[track_id]['speed'] < speed_limit: 
            emoji2_resized = cv2.resize(emoji2, (80, 80))
            overlay_image_alpha(frame, emoji2_resized[:, :, :3], int(l), int(b) + 40, emoji2_resized[:, :, 3] / 255.0)


    cv2.line(frame, line_pts[0], line_pts[1], (255, 0, 0), 2)

    out.write(frame)


cap.release()
out.release()
cv2.destroyAllWindows()