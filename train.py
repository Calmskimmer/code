from ultralytics import YOLO
import torch

# Check for MPS (Metal Performance Shaders) availability
if torch.backends.mps.is_available():
    device = 'mps'  # Use MPS backend for the Apple M1 Pro
else:
    device = 'cpu'  # Fallback to CPU

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')  # YOLOv8 nano model for faster training

# Train the model on your local dataset
model.train(
    data='data.yaml',  # Path to dataset configuration file
    epochs=100,        # Number of training epochs
    imgsz=640,         # Input image size
    batch=16,          # Batch size
    device=device,     # Specify the device (MPS or CPU)
    half=True          # Use mixed precision for faster training (if supported)
)
