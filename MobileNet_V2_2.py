# tf version: 2.13.0 in colab, 2.6.2 in local
# py version: 3.10.12 in colab, 3.6.8 in local

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
from tensorflow.keras.losses import SparseCategoricalCrossentropy

# constants

TRAIN_DATADIR = 'train'
TEST_DATADIR = 'test'
# TRAIN_DATADIR = '/content/Mydrive/MyDrive/Colab Notebooks/CUB200/train'
# TEST_DATADIR = '/content/Mydrive/MyDrive/Colab Notebooks/CUB200/test'
IMG_SIZE = (160, 160)
BATCH_SIZE = 64
VALIDATION_SPLIT = 0.2
SEED = 1339
NUM_CLASSES = 200
FINE_TUNE_EPOCHS = 100
EPOCHS = 10

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

train_dataset = train_dataset.shuffle(buffer_size=10000)

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)


data_augmentation = tf.keras.Sequential([
    RandomCrop(*IMG_SIZE),
    RandomRotation(0.3),
    RandomZoom(0.1, 0.2),
    RandomContrast(0.1),
    RandomTranslation(0.1, 0.1),
    GaussianNoise(0.1),
])

preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input


# rescale pixel values from [0, 255] to [-1, 1]
# the mobilnet expects pixel value in [-1,1]
# but image originally is [0,255]

IMG_SHAPE = IMG_SIZE + (3,) # (160, 160, 3) --> img size 160 x 160, 3 means rgb

# include_top = False --> load a network that doesn't include the classification layers at the top

base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SHAPE,
    include_top=False, # load a network that doesn't include the classification layers at the top
    weights='imagenet'
)

# (1) feature extraction

# freeze convolutional base
# use as a feature extractor
# add classifier on top
# train the top-level classifier

base_model.trainable = False # freeze, and prevent weights being update during training

# base_model.summary()

global_average_layer = GlobalAveragePooling2D()
prediction_layer = Dense(
    NUM_CLASSES, 
    activation = 'softmax',
    kernel_regularizer = regularizers.l2(0.01)
)

inputs = tf.keras.Input(shape=IMG_SHAPE)
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)

x = BatchNormalization()(x)

x = global_average_layer(x)

x = Dropout(0.25)(x)

x = Dense(1024, activation='relu')(x)

x = BatchNormalization()(x)

x = Dropout(0.25)(x)

outputs = prediction_layer(x)

model = tf.keras.Model(inputs, outputs)

# model.summary()

base_learning_rate = 0.0001

model.compile(
    optimizer = Adam(learning_rate = base_learning_rate),
    loss = SparseCategoricalCrossentropy(),
    metrics = ['accuracy']
)

initial_epochs = EPOCHS

loss0, accuracy0 = model.evaluate(validation_dataset)

print("initial loss: {:.2f}".format(loss0))
print("initial accuracy: {:.2f}".format(accuracy0))

history = model.fit(
    train_dataset,
    epochs = initial_epochs,
    validation_data = validation_dataset
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
plt.ylim([min(plt.ylim()),1])
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

base_model.trainable = True

# Let's take a look to see how many layers are in the base model
print("Number of layers in the base model: ", len(base_model.layers))

# Fine-tune from this layer onwards
fine_tune_at = 100

# Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
  layer.trainable = False

model.compile(
    optimizer = Adam(learning_rate = base_learning_rate / 10),
    loss = SparseCategoricalCrossentropy(),
    metrics = ['accuracy']
)

# model.summary()

# len(model.trainable_variables)

total_epochs =  FINE_TUNE_EPOCHS + EPOCHS

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    verbose=1,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.2, 
    patience=5, 
    min_lr=1e-6, 
    verbose=1
)

history_fine = model.fit(
    train_dataset,
    epochs = total_epochs,
    initial_epoch = history.epoch[-1],
    validation_data = validation_dataset,
    callbacks=[early_stopping, reduce_lr]
)

acc += history_fine.history['accuracy']
val_acc += history_fine.history['val_accuracy']

loss += history_fine.history['loss']
val_loss += history_fine.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.ylim([0, 1])
plt.plot([initial_epochs-1,initial_epochs-1],
          plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.ylim([0, max(history.history['loss'])+1])
plt.plot([initial_epochs-1,initial_epochs-1],
         plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()

# 0.569