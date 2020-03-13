import os
import cv2
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
import tensorflow
from tensorflow.keras.utils import plot_model

from tensorflow.keras import models, layers
from tensorflow.keras.models import Model as KerasModel
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Reshape, Concatenate, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Options: EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3
# Higher the number, the more complex the model is.
import efficientnet.tfkeras as efn

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

acc_metric = tensorflow.keras.metrics.Accuracy()
bce_metric = tensorflow.keras.metrics.BinaryCrossentropy()
bce_loss = tensorflow.keras.losses.BinaryCrossentropy()

# Specify image size
IMG_WIDTH = 240
IMG_HEIGHT = 240
CHANNELS = 3

# loading pretrained conv base model
conv_base = efn.EfficientNetB1(weights="imagenet", include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, CHANNELS))

dropout_rate = 0.15
model = models.Sequential()
model.add(conv_base)
model.add(layers.GlobalAveragePooling2D(name="gap"))
model.add(layers.Dropout(dropout_rate, name="dropout_out"))
model.add(layers.Dense(1, activation="sigmoid"))
conv_base.trainable = True

print(model.summary())

print('Loading data files...')
data_path = 'D:/Data/deepfake-detection-challenge/dfdc_processed'
metadata = pd.read_csv(data_path + '/metadata_copy.csv')

chunk_prefix = 'dfdc_train_part_'
all_dirs = [chunk_prefix + str(i).zfill(2) for i in range(50)]
train_dirs = [chunk_prefix + str(i).zfill(2) for i in range(30)]
train_dirs = train_dirs + [chunk_prefix + str(i).zfill(2) for i in range(40, 50)]
# train_dirs = [chunk_prefix + str(i).zfill(2) for i in range(50) if i % 10 != 0]
valid_dirs = [chunk for chunk in all_dirs if chunk not in train_dirs]

train_df = metadata.loc[metadata['chunk'].isin(train_dirs)].reset_index(drop=True)
valid_df = metadata.loc[metadata['chunk'].isin(valid_dirs)].reset_index(drop=True)

# balance validation set
print('Preparing data and model...')
balance_valid_data = True

if balance_valid_data:
    temp_df = pd.DataFrame(columns=valid_df.columns)
    for chunk in valid_dirs:
        chunk_df = valid_df.loc[valid_df['chunk'] == chunk]
        group = chunk_df.groupby('target')
        group = pd.DataFrame(group.apply(lambda x: x.sample(group.size().min()).reset_index(drop=True)))
        temp_df = temp_df.append(group, ignore_index=True)
        del group

    valid_df = temp_df.copy()

print('Validation dataframe contains:')
print(valid_df['label'].value_counts())

image_size = 240
batch_size = 16
train_steps = math.ceil(len(train_df) / batch_size)
valid_steps = math.ceil(len(valid_df) / batch_size)


def preprocess_image(image, sigmaX=10):
    """
    The whole preprocessing pipeline:
    1. Read in image
    2. Apply masks
    3. Resize image to desired size
    4. Add Gaussian noise to increase Robustness

    :param img: A NumPy Array that will be cropped
    :param sigmaX: Value used for add GaussianBlur to the image

    :return: A NumPy array containing the preprocessed image
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)
    return image


learning_rate = 1e-4

# This callback will stop the training when there is no improvement in
# the validation loss for p consecutive epochs.
early_stopping = tensorflow.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    verbose=1)


# This function keeps the learning rate at 0.001 for the first ten epochs
# and decreases it exponentially after that.
def scheduler(epoch):
    if epoch < 2:
        # return learning_rate * math.exp(-0.1 * epoch)
        return learning_rate

    else:
        return learning_rate * 0.2


lr_scheduler = tensorflow.keras.callbacks.LearningRateScheduler(
    scheduler,
    verbose=1)

reduce_lr = tensorflow.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-8,
    verbose=1)

model.compile(
    loss=tensorflow.keras.losses.BinaryCrossentropy(),
    optimizer=tensorflow.keras.optimizers.RMSprop(lr=learning_rate),
    metrics=[tensorflow.keras.metrics.AUC()])

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    channel_shift_range=100.0,
    fill_mode='nearest')

valid_datagen = ImageDataGenerator(
    rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=None,
        x_col="path",
        y_col="label",
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='binary',
        validate_filenames=False)

validation_generator = valid_datagen.flow_from_dataframe(
        dataframe=valid_df,
        directory=None,
        x_col="path",
        y_col="label",
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='binary',
        validate_filenames=False)

print('Starting training...')
model.fit(
        train_generator,
        steps_per_epoch=train_steps,
        epochs=4,
        callbacks=[lr_scheduler],
        validation_data=validation_generator,
        validation_steps=valid_steps,
        verbose=1)

# Plot training & validation accuracy values
history = model.history
plt.plot(history.history['auc'])
plt.plot(history.history['val_auc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# save the model
model.save('D:/Data/deepfake-detection-challenge/checkpoints/model_a2.h5')
