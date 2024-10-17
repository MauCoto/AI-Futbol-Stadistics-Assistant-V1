import cv2
import numpy as np
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
    results = model.predict(source=frame)
    detections = results[0].boxes.data.cpu().numpy()  # Get detections as numpy array

    x1 = y1 = x2 = y2 = None
    confidence = class_id = center_x = center_y = None
    width = height = 0

    for detection in detections:
        x1, y1, x2, y2 = detection[:4]
        confidence = detection[4] if len(detection) > 4 else 0.0
        class_id = detection[5] if len(detection) > 5 else -1
        if confidence > 0.7 and int(class_id) == 0:  # Only consider persons
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            width = x2 - x1
            height = y2 - y1

            # Only draw the bounding box if the confidence is above a higher threshold
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Persona: {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Initialize previous coordinates if not already initialized
            if 'prev_x' not in locals() or 'prev_y' not in locals():
                prev_x, prev_y = center_x, center_y
            
            # Calculate cumulative distance traveled by the person
            distance_px = np.sqrt((center_x - prev_x)**2 + (center_y - prev_y)**2)
            distance_m = distance_px * 0.003  # Assuming 1 pixel = 0.05 meters

            if 'total_distance_m' not in locals():
                total_distance_m = 0
            total_distance_m += distance_m

            time_s = 1 / cap.get(cv2.CAP_PROP_FPS)
            velocity_mps = distance_m / time_s
            velocity_kmph = velocity_mps * 3.6  # Convert m/s to km/h

            # Calculate average velocity
            if 'total_time_s' not in locals():
                total_time_s = 0
            total_time_s += time_s
            average_velocity_mps = total_distance_m / total_time_s
            average_velocity_kmph = average_velocity_mps * 3.6  # Convert m/s to km/h

            cv2.putText(frame, f"Distancia Recorrida: {total_distance_m:.2f} m", (frame.shape[1] // 2 - 200, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Velocidad Promedio: {average_velocity_kmph:.2f} km/h", (frame.shape[1] // 2 - 200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            prev_x, prev_y = center_x, center_y

    cv2.imshow("Video", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
