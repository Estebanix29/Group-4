import tensorflow as tf
import pandas as pd
import keras
from sklearn.model_selection import train_test_split
from keras.src.layers import Dense, LeakyReLU, Dropout
import matplotlib.pyplot as plt
import numpy as np

device_name = tf.test.gpu_device_name()
if device_name == '/device:GPU:0':
    print('Found GPU at: {}'.format(device_name))
else:
    print('No GPU found, using CPU')

print(tf.__version__)

def residual_block(x, filters, kernel_size=3):
    shortcut = x
    x = keras.layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)

    x = keras.layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)

    if x.shape[-1] != shortcut.shape[-1]:
        shortcut = keras.layers.Conv2D(filters, 1)(shortcut)
        shortcut = keras.layers.BatchNormalization()(shortcut)

    x = keras.layers.add([shortcut, x])
    x = keras.layers.Activation('relu')(x)
    return x

inputs = keras.Input(shape=(32, 32, 3))

#Initial Convolutional layer
x = keras.layers.Conv2D(32, (3, 3), padding='same')(inputs)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Activation('relu')(x)

#Redisual blocks
x = residual_block(x, 64)
x = keras.layers.MaxPooling2D((2, 2))(x) #downsample
x = residual_block(x, 128)
#dropout layer
x = keras.layers.Dropout(0.25)(x)

x = residual_block(x, 256)
x = residual_block(x, 512)

#Output and global average
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.25)(x)

# Dense hidden layer
x = keras.layers.Dense(128, activation='relu')(x)
x = keras.layers.Dropout(0.5)(x)

outputs = keras.layers.Dense(10, activation='softmax')(x)

# Create the model
model = keras.Model(inputs, outputs)

model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                metrics=['accuracy'])
