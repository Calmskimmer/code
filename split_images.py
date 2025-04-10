import os
import shutil
import random

def split_dataset(input_folder, output_folder, train_ratio=0.8):
    # Ensure output directory structure
    train_images_path = os.path.join(output_folder, 'train', 'images')
    train_texts_path = os.path.join(output_folder, 'train', 'labels')
    valid_images_path = os.path.join(output_folder, 'valid', 'images')
    valid_texts_path = os.path.join(output_folder, 'valid', 'labels')

    os.makedirs(train_images_path, exist_ok=True)
    os.makedirs(train_texts_path, exist_ok=True)
    os.makedirs(valid_images_path, exist_ok=True)
    os.makedirs(valid_texts_path, exist_ok=True)

    # Collect image and text files
    files = os.listdir(input_folder)
    images = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Shuffle and split
    random.shuffle(images)
    split_index = int(len(images) * train_ratio)
    train_images = images[:split_index]
    valid_images = images[split_index:]

    # Copy files to respective folders
    for image_list, folder in zip([train_images, valid_images], ['train', 'valid']):
        for image in image_list:
            # Get corresponding text file
            text_file = os.path.splitext(image)[0] + '.txt'
            image_src = os.path.join(input_folder, image)
            text_src = os.path.join(input_folder, text_file)

            image_dest = os.path.join(output_folder, folder, 'images', image)
            text_dest = os.path.join(output_folder, folder, 'labels', text_file)

            # Move files if both image and text exist
            if os.path.exists(text_src):
                shutil.copy(image_src, image_dest)
                shutil.copy(text_src, text_dest)

if __name__ == "__main__":
    input_folder = "/Users/micklammers/Documents/Trickshot/data/y"  # Replace with your input folder path
    output_folder = "/Users/micklammers/Documents/Trickshot/data/x"  # Replace with your output folder path
    split_dataset(input_folder, output_folder)
