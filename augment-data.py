"""

Flips images horizontally in order to augment dataset

"""

import os
import cv2

# Horizontal flip function
def mirror_image(image):
    return cv2.flip(image, 1)

# Input images folder
input_dir = 'dataset/comparison/dlc/242-f'

# New folder for processed images
output_dir = os.path.join(input_dir, 'augmented')
os.makedirs(output_dir, exist_ok=True)

# File list
image_files = [f for f in os.listdir(input_dir) if f.endswith('.jpg') or f.endswith('.png')]

# Image processing loop
for image_file in image_files:
    input_path = os.path.join(input_dir, image_file)
    output_path = os.path.join(output_dir, image_file)

    image = cv2.imread(input_path)

    mirrored_image = mirror_image(image)

    cv2.imwrite(output_path, mirrored_image)

print('Images have been processed!')
