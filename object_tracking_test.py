import cv2

# Initialize variables
selected = False
tracking = False
bbox = None  # Bounding box for the selected object
tracker = None

def select_object(event, x, y, flags, param):
    global bbox, selected, tracking, tracker
    if event == cv2.EVENT_LBUTTONDOWN:
        bbox = (x - 50, y - 50, 100, 100)  # Initialize bounding box around tap
        selected = True
        tracking = False  # Reset tracking

# Initialize video capture
cap = cv2.VideoCapture(0)  # Use camera feed
cv2.namedWindow("Tracking")
cv2.setMouseCallback("Tracking", select_object)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if selected:
        # Initialize tracker
        tracker = cv2.TrackerCSRT_create()
        tracker.init(frame, bbox)
        selected = False
        tracking = True

    if tracking:
        # Update the tracker
        success, bbox = tracker.update(frame)
        if success:
            # Draw bounding box
            x, y, w, h = map(int, bbox)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            tracking = False  # Stop tracking if object is lost

    # Display the frame
    cv2.imshow("Tracking", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
