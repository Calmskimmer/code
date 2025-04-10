import cv2
import os

def video_to_frames(video_path, output_folder):
    """
    Extracts frames from a video and saves them as images in the specified folder.

    Args:
    - video_path (str): Path to the video file.
    - output_folder (str): Folder where the frames will be saved.

    Returns:
    - None
    """
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Capture the video
    video = cv2.VideoCapture(video_path)
    
    # Check if the video is loaded
    if not video.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return

    # Frame count
    frame_count = 0

    while True:
        # Read a frame
        ret, frame = video.read()
        
        # Break the loop if no frame is returned
        if not ret:
            break

        # Save the frame as an image
        frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)

        # Print progress
        print(f"Saved: {frame_filename}")
        
        # Increment frame count
        frame_count += 1

    # Release the video capture object
    video.release()
    print(f"Done! Extracted {frame_count} frames to {output_folder}.")

# Example usage
video_to_frames("../IMG_5078.mov", "output/5078frames")
