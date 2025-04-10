import cv2
import numpy as np
import onnxruntime as ort

# Load the ONNX model
ONNX_MODEL_PATH = "ball_best.onnx"
MOV_FILE_PATH = "videos/IMG_5078.mov"

# Initialize the ONNX runtime session
session = ort.InferenceSession(ONNX_MODEL_PATH)

# Get input and output details
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape
output_name = session.get_outputs()[0].name
height, width = input_shape[2], input_shape[3]

# Function to preprocess frames
def preprocess_frame(frame):
    # Resize frame to match the model's input size
    resized_frame = cv2.resize(frame, (width, height))
    # Normalize pixel values to [0, 1] if needed
    normalized_frame = resized_frame / 255.0
    # Transpose dimensions to match ONNX input format (NCHW)
    input_data = np.transpose(normalized_frame, (2, 0, 1)).astype(np.float32)
    # Add batch dimension
    input_data = np.expand_dims(input_data, axis=0)
    return input_data

# Function to draw detections
def draw_detections(frame, detections):
    for detection in detections:
        x1, y1, x2, y2, conf, cls = detection
        # Draw bounding box
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        # Add label
        label = f"{int(cls)}: {conf:.2f}"
        cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Open the video file
cap = cv2.VideoCapture(MOV_FILE_PATH)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    input_data = preprocess_frame(frame)

    # Run ONNX model inference
    outputs = session.run([output_name], {input_name: input_data})

    # Extract detections (assuming output format: [batch, num_detections, 6])
    detections = []
    for detection in outputs[0][0]:
        x1, y1, x2, y2, conf, cls = detection[:6]
        if conf > 0.5:  # Apply confidence threshold
            x1, y1, x2, y2 = x1 * frame.shape[1], y1 * frame.shape[0], x2 * frame.shape[1], y2 * frame.shape[0]
            detections.append((x1, y1, x2, y2, conf, cls))

    # Draw detections on the frame
    draw_detections(frame, detections)

    # Display the frame
    cv2.imshow('ONNX Detection', frame)

    # Break on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
