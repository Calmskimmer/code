from ultralytics import YOLO
import cv2
import json  # Import JSON module

# Path to your model and video file
MODEL_PATH = '/best_float32 (1).tflite'
VIDEO_PATH = 'test_videos/Atta_ball.mp4'

# Load the model for detection
model = YOLO(MODEL_PATH, task='detect')

# Open the video file
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"Error: Could not open video {VIDEO_PATH}")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 inference
    results = model.predict(frame, imgsz=640, conf=0.2, verbose=False)

    # Create a JSON output structure for the current frame
    frame_data = {"frame_id": int(cap.get(cv2.CAP_PROP_POS_FRAMES)), "objects": []}

    # Extract detections
    for result in results:
        print("start")
        #print(dir(result))
        print(dir(result.boxes))

        print("end")
        boxes = result.boxes.xyxy.numpy()  # Bounding box coordinates
        scores = result.boxes.conf.numpy()  # Confidence scores
        classes = result.boxes.cls.numpy()  # Class IDs

        # Add detections to the JSON output
        for box, score, cls in zip(boxes, scores, classes):
            x1, y1, x2, y2 = map(int, box)
            detection_data = {
                "bbox": [x1, y1, x2, y2],
                "confidence": float(score),
                "class_id": int(cls),
            }
            frame_data["objects"].append(detection_data)

            # Draw bounding boxes on the frame
            label = f"{int(cls)}: {score:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Print JSON output to the terminal
    print(json.dumps(frame_data, indent=4))

    # Display the frame live
    cv2.imshow("YOLOv8 Live Inference", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
