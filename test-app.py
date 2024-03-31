import tensorflow as tf
import numpy as np
import cv2
from model import Conv3DModel
import mediapipe as mp

# Available Gestures
classes = [
    "Forward",
    "Back",
    "Turn Right",
    "Turn Left",
    "No Command"
]


def normalize_data(np_data):
    # Transforms video into numpy array neural network usage
    # Shape of a single array is (video count, number of frames, height, width, number of channels)
    scaled_images = np_data.reshape(-1, 30, 64, 64, 1)
    return scaled_images


# Creating instance of a 3D-CNN model
new_model = Conv3DModel()

# Initialization of model
new_model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam())

# Loading weights of a previously trained model
new_model.load_weights('model_weights/variables/variables')

# Gesture Recognition variables
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

    result = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    #uncomment code block below in order to show video without processing
    cv2.namedWindow('Human', cv2.WINDOW_NORMAL)
    cv2.imshow('Human', frame)

    black_image = np.zeros(frame.shape, dtype=np.uint8)
    mp_drawing.draw_landmarks(black_image, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    gray = cv2.cvtColor(black_image, cv2.COLOR_BGR2GRAY)
    to_predict.append(cv2.resize(gray, (64, 64)))

    # Takes 30 frames of gesture for prediction
    if len(to_predict) == 30:
        frame_to_predict = np.array(to_predict, dtype=np.float32)  # Turns frames into numbers
        model_input = normalize_data(frame_to_predict)  # Transforms current array to 5 digit format for model input
        predict = new_model.predict(model_input)  # Prediction itself
        image_class = classes[np.argmax(predict)]  # Some black magic to guess which gesture it is

        print('Classe = ', image_class, 'Precision = ', np.amax(predict) * 100, '%')  # Console log

        to_predict = []  # Releases an array to get another batch of frames

    # Show the result within the frame
    cv2.putText(black_image, image_class, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

    # Display the resulting frame
    cv2.namedWindow('Hand Gesture Recognition', cv2.WINDOW_NORMAL)
    # uncomment code above in order to make window editable
    cv2.imshow('Hand Gesture Recognition', black_image)

    # Wait for any key to quit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
