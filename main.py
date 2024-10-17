import cv2
import numpy as np
import torch
from ultralytics import YOLO
# Load YOLOv model
model = YOLO("yolo11n.pt")

# Load video
cap = cv2.VideoCapture(r"C:\Dta\Proyectos Mau\AI Futbol Stadistics Assistant V1\darknet\data\157A6151.MP4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform detection
    results = model.predict(frame)
    detections = results.xyxy[0].cpu().numpy()  # Get detections as numpy array

    for detection in detections:
        x1, y1, x2, y2, confidence, class_id = detection
        if confidence > 0.5 and int(class_id) == 0:  # Only consider persons
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            width = x2 - x1
            height = y2 - y1

            # Only draw the bounding box if the confidence is above a higher threshold
            if confidence >= 0.99:
                if 'bbox_drawn' not in locals() or not bbox_drawn:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"Person: {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    # Calculate velocity and distance
                    if 'prev_x' in locals() and 'prev_y' in locals():
                        distance_px = np.sqrt((center_x - prev_x)**2 + (center_y - prev_y)**2)
                        distance_m = distance_px * 0.05  # Assuming 1 pixel = 0.05 meters
                        time_s = 1 / cap.get(cv2.CAP_PROP_FPS)
                        velocity_mps = distance_m / time_s
                        velocity_kmph = velocity_mps * 3.6  # Convert m/s to km/h
                        cv2.putText(frame, f"Velocity: {velocity_kmph:.2f} km/h", (frame.shape[1] // 2 - 200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(frame, f"Distance: {distance_m:.2f} m", (frame.shape[1] // 2 - 200, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    prev_x, prev_y = center_x, center_y
                    bbox_drawn = True
                else:
                    bbox_drawn = False

    cv2.imshow("Video", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
