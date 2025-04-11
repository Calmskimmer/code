from ultralytics import YOLO
import cv2
import os

# Path to your model and video file
MODEL_PATH = '/ball_best_float32.tflite'
VIDEO_PATH = '/test_videos/Demo_suitable.mov'
OUTPUT_VIDEO_PATH = 'output_video.mp4'

# Load the model for detection
model = YOLO(MODEL_PATH, task='detect')

# Open the video file
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"Error: Could not open video {VIDEO_PATH}")
    exit()

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for output video

# Create a VideoWriter object
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 inference
    results = model.predict(frame, imgsz=640, conf=0.2, verbose=False)

    # Extract detections from the first frame (since this is a single frame inference)
    for result in results:
        boxes = result.boxes.xyxy.numpy()  # Bounding box coordinates
        scores = result.boxes.conf.numpy()  # Confidence scores
        classes = result.boxes.cls.numpy()  # Class IDs

        # Draw bounding boxes on the frame
        for box, score, cls in zip(boxes, scores, classes):
            x1, y1, x2, y2 = map(int, box)
            label = f"{int(cls)}: {score:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Write the processed frame to the output video
    out.write(frame)

# Release resources
cap.release()
out.release()

print(f"Processed video saved as {OUTPUT_VIDEO_PATH}")
