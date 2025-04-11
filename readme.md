Important files:\
yolo_test.py: to test the yolo model\
run14_640.pt: currently best model on 640x360 image size\


##How to finetune a model\
Step 1: put a new video in the data/videos folder\
step 2: run the extract_screenshot.py script to extract screenshots\
step 3: move the screenshot manually to the output/all folder\
Step 4: Open LabelImg and start labeling a number of frames from the video in the /all folder, make sure to store the labels in the trickshot/labels folder, or ../labels\
Step 5: change the output folder name in the move_images.py script and run the file\
Step 6: Run the split_images.py script with new folder names.\
Step 7: Move the folders train and valid from the output folder of the split_images script to the /data folder (one layer up)\
