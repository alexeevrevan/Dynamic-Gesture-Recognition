'''
    This module is supposed to apply skeletons to extracted frames from videos
    It creates a separate dataset with skeletons applied
'''

import cv2
import mediapipe as mp
import os

# Pose settings for MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

folder_path = 'data'

# Launching pose estimation
with mp_pose.Pose(min_detection_confidence=0.5,
                  min_tracking_confidence=0.5) as pose:
    # Walking through frames
    for filename in os.listdir(folder_path):
        img = cv2.imread(f'{folder_path}/{filename}')  # variable containing an image

        #image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        result = pose.process(img)

        # Drawing skeleton on a frame
        mp_drawing.draw_landmarks(img, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Creating updated data
        cv2.imwrite(f'skeleton/002/{filename}', img)
        print(f'Creating {filename}')

print('Skeletons are created')