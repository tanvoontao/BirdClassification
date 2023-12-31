import os
import pickle
import random

import cv2
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    GlobalAveragePooling2D,
    MaxPooling2D,

    # data augmentation
    RandomContrast,
    RandomCrop,
    RandomFlip,
    RandomRotation,
    RandomTranslation,
    RandomZoom,
    # RandomBrightness,
    Rescaling,
    GaussianNoise,
)
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.optimizers import Adam

# constants
# TRAIN_DATADIR = 'train'
# TEST_DATADIR = 'test'
TRAIN_DATADIR = '/content/Mydrive/MyDrive/Colab Notebooks/CUB200/train'
TEST_DATADIR = '/content/Mydrive/MyDrive/Colab Notebooks/CUB200/test'
IMG_SIZE = (128, 128)
BATCH_SIZE = 64
VALIDATION_SPLIT = 0.2
SEED = 1339
NUM_CLASSES = 200
EPOCHS = 100
AUTOTUNE = tf.data.AUTOTUNE

train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DATADIR,
    shuffle=True,
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    validation_split=VALIDATION_SPLIT,
    subset="training",
    seed=SEED
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DATADIR,
    shuffle=True,
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    validation_split=VALIDATION_SPLIT,
    subset="validation",
    seed=SEED
)

# data augmentation to apply each image
def flip_image(image, label):
    return tf.image.flip_left_right(image), label

def adjust_brightness(image, label):
    delta = tf.random.uniform([], -0.2, 0.2) # Random brightness adjustment in the range [-0.2, 0.2]
    image = tf.image.adjust_brightness(image, delta)
    return image, label

def flip_vertical(image, label):
    return tf.image.flip_up_down(image), label

def adjust_saturation(image, label):
    factor = tf.random.uniform([], 0.8, 1.2) # Randomly adjust saturation in the range [0.8, 1.2]
    image = tf.image.adjust_saturation(image, factor)
    return image, label


# data augmentation, append train dataset, shuffling
flipped_horizontal_dataset = train_dataset.map(flip_image, num_parallel_calls=AUTOTUNE)
flipped_vertical_dataset = train_dataset.map(flip_vertical, num_parallel_calls=AUTOTUNE)
bright_dataset = train_dataset.map(adjust_brightness, num_parallel_calls=AUTOTUNE)
saturation_dataset = train_dataset.map(adjust_saturation, num_parallel_calls=AUTOTUNE)

train_dataset = train_dataset.concatenate(flipped_horizontal_dataset)
train_dataset = train_dataset.concatenate(flipped_vertical_dataset)
train_dataset = train_dataset.concatenate(bright_dataset)
train_dataset = train_dataset.concatenate(saturation_dataset)

# Breaking Sequential Bias, and avoid overfit as it might not see a representative sample of the entire distribution, 
train_dataset = train_dataset.shuffle(buffer_size=10000)


train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)


data_augmentation = tf.keras.Sequential([
    # RandomFlip('horizontal'),
    RandomCrop(*IMG_SIZE),
    RandomRotation(0.3),
    RandomZoom(0.1, 0.2),
    RandomContrast(0.1),
    RandomTranslation(0.1, 0.1),
    GaussianNoise(0.1),
])

IMG_SHAPE = IMG_SIZE + (3,)

inputs = tf.keras.Input(shape=IMG_SHAPE)
x = data_augmentation(inputs)
x = Rescaling(1./255)(x)

# [Input Layer]
x = Conv2D(16, kernel_size=[3, 3], padding='same', activation='relu')(x)
x = Conv2D(32, kernel_size=[3, 3], padding='same', activation='relu')(x)
x = MaxPooling2D(pool_size=[3,3])(x)

# [Hidden Layer] #1
x = Conv2D(32, kernel_size=[3, 3], padding='same', activation='relu')(x)
x = Conv2D(64, kernel_size=[3, 3], padding='same', activation='relu')(x)
x = MaxPooling2D(pool_size=[3,3])(x)

# [Hidden Layer] #2
x = Conv2D(128, kernel_size=[3, 3], padding='same', activation='relu')(x)
x = Conv2D(256, kernel_size=[3, 3], padding='same', activation='relu')(x)
x = MaxPooling2D(pool_size=[3,3])(x)

x = BatchNormalization()(x)

# [Fully Connected Layer]
x = Flatten()(x) # convert 3D feature maps to 1D feature vector
x = Dropout(0.5)(x)
x = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
outputs = Dense(NUM_CLASSES, activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()


early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    verbose=1,
    restore_best_weights=True
)

history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=validation_dataset,
    callbacks=[early_stopping]
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()), 1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0, max(history.history['loss'])+1])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()