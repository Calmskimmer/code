import cv2
import numpy as np

# Load video or live camera feed
cap = cv2.VideoCapture('videos/IMG_5078.mov')  # Replace with 0 for webcam

# Store initial frames for background estimation
frames_to_sample = 300  # Number of frames to calculate the median background
frame_list = []

# Read frames for background estimation
print("Capturing background frames...")
for i in range(frames_to_sample):
    ret, frame = cap.read()
    if not ret:
        break
    frame_list.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))  # Convert to grayscale

# Compute the median background
median_background = np.median(np.array(frame_list), axis=0).astype(dtype=np.uint8)

print("Background model computed. Starting background removal...")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert current frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Background subtraction
    foreground_mask = cv2.absdiff(gray_frame, median_background)

    # Apply threshold to extract foreground objects
    _, thresh = cv2.threshold(foreground_mask, 30, 255, cv2.THRESH_BINARY)

    # Optional: Apply morphological operations to reduce noise
    kernel = np.ones((5, 5), np.uint8)
    clean_thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # Find contours (optional visualization)
    contours, _ = cv2.findContours(clean_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result_frame = frame.copy()
    cv2.drawContours(result_frame, contours, -1, (0, 255, 0), 2)

    # Show results
    cv2.imshow("Original Frame", frame)
    cv2.imshow("Foreground Mask", clean_thresh)
    cv2.imshow("Detected Objects", result_frame)

    # Exit on pressing 'q'
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
