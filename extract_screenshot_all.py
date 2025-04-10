import cv2
import os

# Define the directory containing video files and the output directory
video_dir = '/Users/micklammers/Documents/Trickshot/data/videos'
output_dir = '/Users/micklammers/Documents/Trickshot/data/output/all_frames2'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Get all video files that start with 'vid'
video_paths = [
    os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.startswith('*vid')
]

frame_count = 0

for video_path in video_paths:
    # Extract the video name (without extension) from the video path
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # Open the video file
    video_capture = cv2.VideoCapture(video_path)
    
    # Check if the video file was successfully opened
    if not video_capture.isOpened():
        print(f"Error: Could not open video {video_path}")
        continue
    
    frame_index = 0  # Track frame index
    
    # Loop through the video and extract frames
    while True:
        # Read the next frame
        success, frame = video_capture.read()
        
        if not success:
            break
        
        # Skip every other frame
        if frame_index % 2 == 0:
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
        
        frame_index += 1  # Increment frame index
    
    # Release the video capture object
    video_capture.release()

print(f"Finished extracting frames. Total frames saved: {frame_count}")
