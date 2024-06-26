"""

3D CNN Model container

"""

import tensorflow as tf


class Conv3DModel(tf.keras.Model):
    def __init__(self):
        super(Conv3DModel, self).__init__()
        # Convolutions
        self.conv1 = tf.compat.v2.keras.layers.Conv3D(32, (3, 3, 3), activation='relu', name="conv1",
                                                      data_format='channels_last')
        self.batch_norm1 = tf.keras.layers.BatchNormalization(momentum=0.99)
        self.pool1 = tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2), data_format='channels_last')
        self.dropout1 = tf.keras.layers.Dropout(0.25)

        self.conv2 = tf.compat.v2.keras.layers.Conv3D(64, (3, 3, 3), activation='relu', name="conv1",
                                                      data_format='channels_last')
        self.batch_norm2 = tf.keras.layers.BatchNormalization(momentum=0.99)
        self.pool2 = tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2), data_format='channels_last')
        self.dropout2 = tf.keras.layers.Dropout(0.25)

        # LSTM & Flatten
        self.convLSTM = tf.keras.layers.ConvLSTM2D(40, (3, 3))
        self.flatten = tf.keras.layers.Flatten(name="flatten")

        # Dense layers
        self.d1 = tf.keras.layers.Dense(128, activation='relu', name="d1")
        self.batch_norm3 = tf.keras.layers.BatchNormalization(momentum=0.99)
        self.dropout3 = tf.keras.layers.Dropout(0.5)
        self.out = tf.keras.layers.Dense(5, activation='softmax', name="output")

    def call(self, x):
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        x = self.convLSTM(x)
        x = self.flatten(x)
        x = self.d1(x)
        x = self.batch_norm3(x)
        x = self.dropout3(x)
        return self.out(x)