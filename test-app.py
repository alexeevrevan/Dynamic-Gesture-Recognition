import tensorflow as tf
import numpy as np
import cv2
from model import Conv3DModel
import mediapipe as mp

# Available Gestures
classes = [
    "Move Forward",
    "Move Back",
    "Turn Right",
    "Turn Left",
    "Full Stop",
    "No Command"
]


def normalize_data(np_data):
    # Transforms existing array into a format of (don't know, batch size, height, width, channels)
    scaled_images = np_data.reshape(-1, 30, 64, 64, 1)
    return scaled_images


# Creating instance of model
new_model = Conv3DModel()

# Initialization of model
new_model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=tf.keras.optimizers.legacy.Adam())

# Loading weights from trained model
new_model.load_weights('model_weights/cp-0007.ckpt/variables/variables')

# Gesture Recognition
to_predict = []
cap = cv2.VideoCapture(0)
image_class = ''

# Pose settings for MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5,
                  min_tracking_confidence=0.5)

while (True):
    # Camera capture frame-by-frame
    ret, frame = cap.read()

    # That's where MediaPipe does its processing
    result = pose.process(frame)
    mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # This variable contains the frame and turns it to gray

    to_predict.append(cv2.resize(gray, (64, 64)))  # Appends gray frames to an array for further gesture prediction

    # Takes 30 frames of gesture for prediction
    if len(to_predict) == 30:
        frame_to_predict = np.array(to_predict, dtype=np.float32)  # Turns frames into numbers
        model_input = normalize_data(frame_to_predict)  # Transforms current array to 5 digit format for model input
        predict = new_model.predict(model_input)  # Prediction itself
        image_class = classes[np.argmax(predict)]  # Some black magic to guess which gesture it is

        print('Classe = ', image_class, 'Precision = ', np.amax(predict) * 100, '%')  # Console log

        to_predict = []  # Releases an array to get another batch of frames

    # Show the result within the frame
    cv2.putText(frame, image_class, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 110, 50), 1, cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow('Hand Gesture Recognition', frame)

    # Wait for any key to quit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
