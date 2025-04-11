from ultralytics import YOLO

# Load the YOLO model (replace with your model path if needed)
model = YOLO('best.pt')  # or 'yolov8n.pt'

# Print the model's default image size
print("Default input size(s):", model.model.args['imgsz'] if hasattr(model.model, 'args') else model.model.input_size)
