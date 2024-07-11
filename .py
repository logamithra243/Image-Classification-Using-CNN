from tensorflow import keras
import tensorflow as tf
import os,datetime
import tensorflow_datasets as tfds
df, info = tfds.load('rock_paper_scissors', with_info = True, as_supervised = True)

num_validation = 0.1 * info.splits['train'].num_examples
num_validation = tf.cast(num_validation, tf.int64)

train_data = df['train']
test_images = df['test']


def preprocess(image, labels):
  image = tf.cast(image, tf.float32)
  image /= 255.
  return image,labels 
valid_train_data = train_data.map(preprocess)
test_images = test_images.map(preprocess)

valid_train_data_shuffled = valid_train_data.shuffle(1000)
valid_data = valid_train_data_shuffled.take(num_validation)
train_data = valid_train_data_shuffled.skip(num_validation)
num_test = info.splits['test'].num_examples
num_test = tf.cast(num_test, tf.int64)

from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1,  horizontal_flip=True)
it_gen = datagen.flow(train_images, train_labels, batch_size=32)

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, BatchNormalization, Dropout
from tensorflow.keras import layers
from tensorflow.keras import optimizers
import numpy as np

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, BatchNormalization, Dropout
from tensorflow.keras import layers
from tensorflow.keras import optimizers
import numpy as np


model = Sequential([
                    Conv2D(32, 3, padding='same',  activation='relu',kernel_initializer='he_uniform', input_shape = [150, 150, 3]),
                    MaxPooling2D(2),
                    Conv2D(32, 3, padding='same', kernel_initializer='he_uniform', activation='relu'),
                    MaxPooling2D(2),
                    Conv2D(64, 3, padding='same', kernel_initializer='he_uniform',activation='relu'),
                    MaxPooling2D(2),
                    Conv2D(64, 3, padding='same', kernel_initializer='he_uniform',activation='relu'),
                    MaxPooling2D(2),
                    Conv2D(128, 3, padding='same', kernel_initializer='he_uniform',activation='relu'),
                    MaxPooling2D(2),
                    Flatten(),
                    Dense(128, kernel_initializer='he_uniform',activation = 'relu'),
                    Dense(3, activation = 'softmax'),
                    ])

model.summary()

steps = int(train_images.shape[0] / 32)
model.compile(optimizer= optimizers.Adam(1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
early_stopping_cb = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
history = model.fit( it_gen, epochs = 200, validation_data = (valid_images, valid_labels),
                    callbacks = [early_stopping_cb], steps_per_epoch=steps, verbose=2)

import pandas as pd
import matplotlib.pyplot as plt
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0,5) 
plt.show()

model.evaluate(test_images, test_labels)

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
image = train_images[0].reshape((150,150,3))plt.imshow(image, cmap='Accent_r')
