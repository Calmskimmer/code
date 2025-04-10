import cv2
import numpy as np

# Load video or webcam (set to 0 for live camera)
cap = cv2.VideoCapture('videos/IMG_5078.mov')  # Change to 0 for real-time

# Resize for mobile processing (lower resolution = faster)
FRAME_WIDTH = 320
FRAME_HEIGHT = 240

# Define the ball color range (tweak for different colors)
# Example: Detecting an **orange ball**
lower_bound = np.array([5, 100, 100])  # Lower HSV bound
upper_bound = np.array([20, 255, 255])  # Upper HSV bound

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Resize for speed optimization
    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

    # Convert to HSV for color filtering
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Apply color mask to detect the ball
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # Morphological operations to reduce noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Find contours (faster than HoughCircles)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw detected ball(s)
    result_frame = frame.copy()
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 100:  # Filter out small noise
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            center = (int(x), int(y))
            radius = int(radius)
            cv2.circle(result_frame, center, radius, (0, 255, 0), 2)

    # Show the processed frames
    cv2.imshow("Detected Ball", result_frame)
    cv2.imshow("Mask", mask)

    # Press 'q' to exit
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
