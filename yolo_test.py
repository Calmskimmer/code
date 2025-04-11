import cv2
import numpy as np
import json
from ultralytics import YOLO
from sort import Sort  # Ensure sort.py is in the same directory

# Load the YOLOv8 .pt model
YOLO_MODEL_PATH = "run14_640.pt"
MOV_FILE_PATH = "/test_videos/vid3_school_short.mov"

# Initialize the YOLOv8 model
model = YOLO(YOLO_MODEL_PATH)

# Initialize SORT tracker with tuned parameters
tracker = Sort(max_age=30, min_hits=5, iou_threshold=0.3)

# Function to draw tracked detections
def draw_detections(frame, tracked_objects):
    for obj in tracked_objects:
        x1, y1, x2, y2, obj_id = obj
        # Draw bounding box
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        # Add label with object ID
        label = f"ID {int(obj_id)}"
        cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Open the video file
cap = cv2.VideoCapture(MOV_FILE_PATH)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# JSON output structure
output_data = []

frame_id = 0  # Frame counter
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 inference
    results = model.predict(frame)

    # Extract detections in format [x1, y1, x2, y2, confidence]
    detections = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].numpy()
        conf = box.conf[0].item()
        # Add detection if confidence is above a threshold
        if conf > 0.3:
            detections.append([x1, y1, x2, y2, conf])

    # Convert detections to a numpy array
    if len(detections) > 0:
        detections = np.array(detections)
    else:
        detections = np.empty((0, 5))  # Empty array with shape (0, 5)

    # Update tracker with current frame detections
    tracked_objects = tracker.update(detections)

    # Collect data for JSON output
    frame_data = {
        "frame_id": frame_id,
        "objects": []
    }
    for obj in tracked_objects:
        x1, y1, x2, y2, obj_id = obj
        frame_data["objects"].append({
            "id": int(obj_id),
            "bbox": [int(x1), int(y1), int(x2), int(y2)]
        })
    output_data.append(frame_data)

    # Draw tracked objects on the frame
    draw_detections(frame, tracked_objects)

    # Display the frame
    cv2.imshow('YOLOv8 + SORT Tracking', frame)

    # Break on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_id += 1

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()

# Save the collected data to a JSON file
output_json_path = "/Users/micklammers/Documents/Trickshot/data/tracking_results.json"
with open(output_json_path, 'w') as f:
    json.dump(output_data, f, indent=4)

print(f"Tracking results saved to {output_json_path}")
