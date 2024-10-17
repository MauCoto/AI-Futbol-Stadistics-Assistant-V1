import cv2
import numpy as np
from ultralytics import YOLO
from scipy.spatial import distance

# Load YOLOv model
model = YOLO("yolo11n.pt")

# Load video
cap = cv2.VideoCapture(r"C:\Dta\Proyectos Mau\AI Futbol Stadistics Assistant V1\darknet\data\157A6151.MP4")

person_data = {}
next_person_id = 1

def get_closest_person_id(center_x, center_y, person_data, threshold=50):
    min_dist = float('inf')
    closest_person_id = None
    for person_id, data in person_data.items():
        dist = distance.euclidean((center_x, center_y), (data['prev_x'], data['prev_y']))
        if dist < min_dist and dist < threshold:
            min_dist = dist
            closest_person_id = person_id
    return closest_person_id

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform detection
    results = model.predict(source=frame)
    detections = results[0].boxes.data.cpu().numpy()  # Get detections as numpy array

    current_frame_person_data = {}

    for detection in detections:
        x1, y1, x2, y2 = detection[:4]
        confidence = detection[4] if len(detection) > 4 else 0.0
        class_id = detection[5] if len(detection) > 5 else -1
        if confidence > 0.6 and int(class_id) == 0:  # Only consider persons
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            person_id = get_closest_person_id(center_x, center_y, person_data)
            if person_id is None:
                person_id = next_person_id
                next_person_id += 1

            prev_data = person_data.get(person_id, {'prev_x': center_x, 'prev_y': center_y, 'total_distance_m': 0, 'total_time_s': 0})
            distance_px = np.sqrt((center_x - prev_data['prev_x'])**2 + (center_y - prev_data['prev_y'])**2)
            distance_m = distance_px * 0.003  # Assuming 1 pixel = 0.003 meters

            time_s = 1 / cap.get(cv2.CAP_PROP_FPS)
            velocity_mps = distance_m / time_s
            velocity_kmph = velocity_mps * 3.6  # Convert m/s to km/h

            # Calculate cumulative distance traveled by the person
            total_distance_m = prev_data['total_distance_m'] + distance_m

            # Calculate average velocity
            total_time_s = prev_data['total_time_s'] + time_s
            average_velocity_mps = total_distance_m / total_time_s
            average_velocity_kmph = average_velocity_mps * 3.6  # Convert m/s to km/h

            current_frame_person_data[person_id] = {
                'prev_x': center_x,
                'prev_y': center_y,
                'total_distance_m': total_distance_m,
                'total_time_s': total_time_s
            }

            # Only draw the bounding box if the confidence is above a higher threshold
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Persona: {person_id} CF: {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Draw distance and velocity below each person
            cv2.putText(frame, f"Distancia: {total_distance_m:.2f} m", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, f"Velocidad: {average_velocity_kmph:.2f} km/h", (x1, y2 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    person_data = current_frame_person_data

    cv2.imshow("Video", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
