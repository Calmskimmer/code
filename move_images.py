import os
import shutil

def copy_matching_files(text_folder, image_folder, destination_folder):
    """
    Copies all text files and their corresponding images (with the same base name after removing the first character) 
    from the source folders to the destination folder.

    Args:
        text_folder (str): Path to the folder containing .txt files.
        image_folder (str): Path to the folder containing image files.
        destination_folder (str): Path to the destination folder.
    """
    # Create destination folder if it doesn't exist
    os.makedirs(destination_folder, exist_ok=True)

    # Get all .txt files in the text folder
    text_files = [f for f in os.listdir(text_folder) if f.endswith('.txt')]

    # Get all image files in the image folder
    image_files = os.listdir(image_folder)
    
    # Create a dictionary mapping modified image names to their original names
    modified_image_map = {}
    for image_file in image_files:
        image_base_name, image_ext = os.path.splitext(image_file)
        modified_image_base_name = image_base_name[1:] if len(image_base_name) > 1 else image_base_name
        modified_image_map[modified_image_base_name] = image_file
    
    # Iterate over text files
    for text_file in text_files:
        text_base_name = os.path.splitext(text_file)[0]  # Get the base name (without extension)
        modified_text_base_name = text_base_name[1:] if len(text_base_name) > 1 else text_base_name  # Remove first character

        # Find the corresponding image in the modified image map
        if modified_text_base_name in modified_image_map:
            # Copy the text file
            shutil.copy(os.path.join(text_folder, text_file), destination_folder)

            # Copy the corresponding image file
            shutil.copy(os.path.join(image_folder, modified_image_map[modified_text_base_name]), destination_folder)
            print(f"Copied: {text_file} and {modified_image_map[modified_text_base_name]}")

# Example usage
text_folder = "img_lbels/text2"  # Path to folder containing .txt files
image_folder = "output/all_frames2"  # Path to folder containing images
destination_folder = "y"  # Path to destination folder

copy_matching_files(text_folder, image_folder, destination_folder)
