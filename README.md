# Dynamic-Gesture-Recognition
This repository contains a project for a master degree research.
The aim of the research is to create an AI based system of spatio-temporal user gesture recognition.
The 3D-CNN Model used in this case is trained on a custom dataset consisting of multiple videos collected from volunteers

### Technology Stack:
- Python 3.8
- TensorFlow 2.13.0
- Keras 2.13.1
- OpenCV 4.8.1
- MediaPipe 0.10.5


### Modules description:
- **model.py**: contains 3D-CNN model built with Keras API

- **train-app.py**: Main module for training the 3D-CNN model, saving training results and data visualisation

- **test-app.py**: Module for testing the trained module in realtime gesture classification

- **video-to-frames.py**: Video processing module for extracting frames form video 

- **skeleton-to-image.py**: Frame processing module for applying skeleton to image.

- **augment-data.py**: Horizontal flip of a video in order to augment dataset

- **filter-one**: Outlier detector which excludes skeletons which lack main landmarks

- **filter-two**: Outlier detector, excludes gestures which are out of expected range of performance
