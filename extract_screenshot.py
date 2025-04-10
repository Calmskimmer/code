import cv2
import os

# Define the path to the video file and the output directory
video_path = '/Users/micklammers/Documents/Trickshot/data/videos/IMG_5173.MOV'  # Update with the path to your video
output_dir = '/Users/micklammers/Documents/Trickshot/data/output/IMG_5173'

# Extract the video name (without extension) from the video path
video_name = os.path.splitext(os.path.basename(video_path))[0]

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Open the video file
video_capture = cv2.VideoCapture(video_path)

# Check if the video file was successfully opened
if not video_capture.isOpened():
    print(f"Error: Could not open video {video_path}")
    exit()

# Get the total number of frames in the video
total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

# Loop through the video and extract frames
frame_count = 0
while True:
    # Read the next frame
    success, frame = video_capture.read()
    
    if not success:
        break
    
    # Get original dimensions
    height, width = frame.shape[:2]
    
    # Calculate new dimensions while maintaining aspect ratio
    max_size = 640
    if width > height:
        new_width = max_size
        new_height = int((max_size / width) * height)
    else:
        new_height = max_size
        new_width = int((max_size / height) * width)
    
    # Resize the frame
    resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    # Construct the output file path with video name as prefix
    output_file = os.path.join(output_dir, f"{video_name}_frame_{frame_count:04d}.jpg")
    
    # Save the resized frame as an image file
    cv2.imwrite(output_file, resized_frame)
    print(f"Saved {output_file}")
    
    frame_count += 1

# Release the video capture object
video_capture.release()
print(f"Finished extracting frames. Total frames saved: {frame_count}")
