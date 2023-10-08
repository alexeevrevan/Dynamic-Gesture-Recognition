'''
This module is set to train 3D CNN model and save weights for further usage in test-app
'''

# In[1]
import numpy as np
import cv2
import tensorflow as tf
import os
import math
import pandas as pd
import matplotlib.image as img
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import csv
from model import Conv3DModel

# In[2]

# Setting gestures labels
gestures = [
    "Move Forward",
    "Move Back",
    "Turn Right",
    "Turn Left",
    "Full Stop",
    "No Command"
]

# In[3]

# SKIP IF CREATED

# Folder creation script for the frames
path = 'dataset/'

for i in range(60):
    new_folder = str(i)
    if not os.path.exists(f'{path}{new_folder}'):
        os.makedirs(f'dataset/{new_folder}')
        print(f'Created folder {new_folder}.')
    else:
        print(f'Folder {new_folder} already exists')

# In[4]

# SKIP IF CREATED

# Script to create a labeled csv file
list_iterator = 0

with open('dataset/labels.csv', mode='w', newline='') as csvfile:
    fieldnames = ['label', 'id']
    writer = csv.DictWriter(csvfile, delimiter=";", fieldnames=fieldnames)
    writer.writeheader()

    # amount of test gestures is 60
    for i in range(60):
        writer.writerow({"label": str(gestures[list_iterator]), "id": str(i)})
        list_iterator += 1
        if list_iterator == 6:
            list_iterator = 0

# In[5]

temporal_variable = pd.read_csv('dataset/train.csv', header=None, sep=';').to_dict()
training_targets_dict = temporal_variable[0]
temporal_variable = pd.read_csv('dataset/validation.csv', header=None, sep=';').to_dict()
validation_targets_dict = temporal_variable[0]
temporal_variable = None

print(training_targets_dict[0])
print(validation_targets_dict[0])

print(len(training_targets_dict))
print(len(validation_targets_dict))

# In[6]

# Setting up directories of data
training_path = 'dataset/train/'
validation_path = 'dataset/validation/'

training_directories = os.listdir(training_path)
validation_directories = os.listdir(validation_path)

print(len(training_directories))
print(len(validation_directories))

# In[7]

target_frame_number = 30  # Get 30 frames from each folder


# Get equal frame count in each folder
def unify_frames(path):
    frames = os.listdir(path)
    frames_count = len(frames)
    if target_frame_number > frames_count:
        # if folder contains fewer frames than needed, it duplicates last frame
        frames += [frames[-1]] * (target_frame_number - frames_count)
    elif target_frame_number < frames_count:
        # if there's too many frames in the folder, it just takes 30 of them
        frames = frames[0:target_frame_number]

    return frames


# In[8]

# Resizing frames

def resize_frame(frame):
    frame = img.imread(frame)
    frame = cv2.resize(frame, (64, 64))
    return frame


# In[9]

# Frame processing to return gray image
def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


# In[10]

# Adjusting training data

counter_training = 0  # number for training, will be used in further modules
new_frames_training = []  # contains training data
training_targets = []  # array compare frames to labels

for directory in training_directories:
    new_frame = []

    frames = unify_frames(training_path + directory)
    if len(frames) == target_frame_number:
        for frame in frames:
            frame = resize_frame(training_path + directory + '/' + frame)
            new_frame.append(rgb2gray(frame))
            if len(new_frame) == 15:
                new_frames_training.append(new_frame)
                training_targets.append(gestures.index(training_targets_dict[int(directory)]))
                counter_training += 1
                new_frame = []

# In[11]

# Adjust validation data

counter_validation = 0
new_frames_validation = []
validation_targets = []

for directory in validation_directories:
    new_frame = []
    frames = unify_frames(validation_path + directory)

    if len(frames) == target_frame_number:
        for frame in frames:
            frame = resize_frame(validation_path + directory + '/' + frame)
            new_frame.append(rgb2gray(frame))
            if len(new_frame) == 15:
                new_frames_validation.append(new_frame)
                validation_targets.append(gestures.index(validation_targets_dict[int(directory)]))
                counter_validation += 1
                new_frame = []

# In[12]

# Correctness check
print(len(new_frames_training))
print(len(training_targets))

training_targets[0:20]

print(len(new_frames_validation))
print(len(validation_targets))
validation_targets[0:20]

# In[13]

# Visualize the data in the current state
fig = plt.figure()

for i in range(2, 4):
    for num, frame in enumerate(new_frames_training[i][0:18]):
        y = fig.add_subplot(4, 5, num + 1)
        y.imshow(frame, cmap='gray')

    fig = plt.figure()

plt.show()


# In[14]

# RAM cleansing

def release_list(array):
    del array[:]
    del array


# In[15]

# Convert training data to numbers
training_data = np.array(new_frames_training[0:counter_training], dtype=np.float32)

# In[16]

# Convert validation data to numbers
validation_data = np.array(new_frames_validation[0:counter_validation], dtype=np.float32)

# In[17]

# Releasing RAM
release_list(new_frames_validation)
release_list(new_frames_training)


# In[18]

training_data.shape

# In[19]
validation_data.shape

# In[20]

# Training data normalization
print('old mean', training_data.mean())
scaler = StandardScaler()
scaled_images = scaler.fit_transform(training_data.reshape(-1, 15 * 64 * 64))
print('new mean', scaled_images.mean())
scaled_images_training = scaled_images.reshape(-1, 15, 64, 64, 1)
print(scaled_images_training.shape)

# In[21]

# Validation data normalization

print('old mean', validation_data.mean())
scaler = StandardScaler()
scaled_images_two = scaler.fit_transform(validation_data.reshape(-1, 15*64*64))
print('new mean', scaled_images_two.mean())
scaled_images_validation = scaled_images_two.reshape(-1, 15, 64, 64, 1)
print(scaled_images_validation.shape)

# In[22]

# Creating instance of 3D CNN model
model = Conv3DModel()

# In[23]
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

# Setting up model
model.compile(loss=loss_fn,
              optimizer=tf.keras.optimizers.Adam(),
              metrics = ['accuracy'])

# In[24]

# Finalizing both train and validation data
x_train = np.array(scaled_images_training) # Input train data
y_train = np.array(training_targets) # Target train data
x_val = np.array(scaled_images_validation) # Input validation set
y_val = np.array(validation_targets) # Target validation set

# In[25]

print(x_train.shape)
print(y_train.shape)
print(x_val.shape)
print(y_val.shape)

# In[26]

checkpoint_path = 'model_weights/cp-{epoch:04d}.ckpt'
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor="val_accuracy", verbose=1, save_best_only=True)

# In[27]

# Well, let's start training
history = model.fit(x_train, y_train,
                    callbacks = [cp_callback],
                    validation_data=(x_val, y_val),
                    batch_size=64,
                    epochs=150)

# In[28]


# Save all metrics

import json

metrics = history.history

train_loss = metrics['loss']
train_accuracy = metrics['accuracy']
val_loss = metrics['val_loss']
val_accuracy = metrics['val_accuracy']


with open("model_weights/metrics_3.json", "w+") as f:
    json.dump({'train_loss' : train_loss,
               'validation_loss': val_loss,
               'train_accuracy': train_accuracy,
               'validation_accuracy': val_accuracy}, f)


# In[28]


from numpy import arange

# Data visualisation for loss function
epochs = range(1, len(train_loss) + 1)

plt.plot(epochs, train_loss, label='Training Loss')
plt.plot(epochs, val_loss, label='Validation Loss')

plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')

# Set the tick locations
plt.xticks(arange(0, len(train_loss) + 1, 10))

# Display the plot
plt.legend(loc='best')
plt.show()

# In[29]

# Data visualisation for accuracies
epochs = range(1, len(train_accuracy) + 1)

plt.plot(epochs, train_accuracy, label='Training Accuracy')
plt.plot(epochs, val_accuracy, label='Validation Accuracy')

plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

# Set the tick locations
plt.xticks(arange(0, len(train_accuracy) + 1, 10))

# Display the plot
plt.legend(loc='best')
plt.show()
