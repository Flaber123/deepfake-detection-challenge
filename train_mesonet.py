import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
import tensorflow
from tensorflow.keras.models import Model as KerasModel
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Reshape, Concatenate, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

acc_metric = tensorflow.keras.metrics.Accuracy()
bce_metric = tensorflow.keras.metrics.BinaryCrossentropy()
bce_loss = tensorflow.keras.losses.BinaryCrossentropy()


class MesoInception4:
    def __init__(self, image_width, learning_rate=0.001):
        # initialize model structure
        x = Input(shape=(image_width, image_width, 3))
        x1 = self.inception_layer(1, 4, 4, 2)(x)
        x1 = BatchNormalization()(x1)
        x1 = MaxPooling2D(pool_size=(2, 2), padding='same')(x1)

        x2 = self.inception_layer(2, 4, 4, 2)(x1)
        x2 = BatchNormalization()(x2)
        x2 = MaxPooling2D(pool_size=(2, 2), padding='same')(x2)

        x3 = Conv2D(16, (5, 5), padding='same', activation='relu')(x2)
        x3 = BatchNormalization()(x3)
        x3 = MaxPooling2D(pool_size=(2, 2), padding='same')(x3)

        x4 = Conv2D(16, (5, 5), padding='same', activation='relu')(x3)
        x4 = BatchNormalization()(x4)
        x4 = MaxPooling2D(pool_size=(4, 4), padding='same')(x4)

        y = Flatten()(x4)
        y = Dropout(0.5)(y)
        y = Dense(16)(y)
        y = LeakyReLU(alpha=0.1)(y)
        y = Dropout(0.5)(y)
        y = Dense(1, activation='sigmoid')(y)

        self.model = KerasModel(inputs=x, outputs=y)
        self.model.compile(optimizer=Adam(lr=learning_rate),
                           loss=tensorflow.keras.losses.BinaryCrossentropy(),
                           metrics=[tensorflow.keras.metrics.AUC()])

    @staticmethod
    def inception_layer(a, b, c, d):
        def layer(x):
            x1 = Conv2D(a, (1, 1), padding='same', activation='relu')(x)

            x2 = Conv2D(b, (1, 1), padding='same', activation='relu')(x)
            x2 = Conv2D(b, (3, 3), padding='same', activation='relu')(x2)

            x3 = Conv2D(c, (1, 1), padding='same', activation='relu')(x)
            x3 = Conv2D(c, (3, 3), dilation_rate=2, strides=1, padding='same', activation='relu')(x3)

            x4 = Conv2D(d, (1, 1), padding='same', activation='relu')(x)
            x4 = Conv2D(d, (3, 3), dilation_rate=3, strides=1, padding='same', activation='relu')(x4)

            y = Concatenate(axis=-1)([x1, x2, x3, x4])

            return y

        return layer

    def predict(self, x):
        return self.model.predict(x)

    def fit(self, x, y):
        return self.model.train_on_batch(x, y)

    def get_accuracy(self, x, y):
        return self.model.test_on_batch(x, y)

    def load(self, path):
        self.model.load_weights(path)


print('Loading data files...')
data_path = 'D:/Data/deepfake-detection-challenge/dfdc_processed'
metadata = pd.read_csv(data_path + '/metadata_copy.csv')

chunk_prefix = 'dfdc_train_part_'
all_dirs = [chunk_prefix + str(i).zfill(2) for i in range(50)]
train_dirs = [chunk_prefix + str(i).zfill(2) for i in range(30)]
train_dirs = train_dirs + [chunk_prefix + str(i).zfill(2) for i in range(40, 50)]
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
batch_size = 128
train_steps = math.ceil(len(train_df) / batch_size)
valid_steps = math.ceil(len(valid_df) / batch_size)

meso_inception = MesoInception4(image_size, learning_rate=0.001)

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
meso_inception.model.fit(
        train_generator,
        steps_per_epoch=train_steps,
        epochs=5,
        validation_data=validation_generator,
        validation_steps=valid_steps,
        verbose=1)


# Plot training & validation accuracy values
history = meso_inception.model.history
plt.plot(history.history['binary_crossentropy'])
plt.plot(history.history['val_binary_crossentropy'])
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
meso_inception.model.save('D:/Data/deepfake-detection-challenge/checkpoints/mesonet_a0.h5')
