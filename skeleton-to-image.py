'''
    This module is supposed to apply skeletons to extracted frames from videos
    It creates a separate dataset with extracted skeletons on black background
'''

import cv2
import mediapipe as mp
import os
import numpy as np

# Pose settings for MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

folder_path = 'dataset/'

black_background = np.zeros((480, 640, 3), dtype=np.uint8)

# Launching pose estimation
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.2) as pose:
    # Walking through frames
    for filename in os.listdir(folder_path):
        img = cv2.imread(f'{folder_path}/{filename}')  # variable containing an image

        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        # Drawing skeleton on an empty background
        black_background.fill(0)
        mp_drawing.draw_landmarks(black_background, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Creating updated data
        cv2.imwrite(f'skeleton/002/{filename}', black_background)
        print(f'Creating {filename}')

print('Skeletons are created')