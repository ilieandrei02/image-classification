import math
import warnings

import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop

import plot_stats
import split_data
import os

DATASET_DIR = "./dataset"
EPOCHS = 10
VALIDATION_SPLIT = 0.2
IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64
BATCH_SIZE = 32

split_data.split_data(DATASET_DIR, "Dog", VALIDATION_SPLIT)
split_data.split_data(DATASET_DIR, "Cat", VALIDATION_SPLIT)


def create_model():
    """Creates a CNN with 4 convolutional layers"""
    return tf.keras.models.Sequential([
        tf.keras.Input(shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3)),  # Input layer
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(len(class_names), activation='softmax')

        # tf.keras.layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
        # tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        # tf.keras.layers.MaxPooling2D(2, 2),
        # tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        # tf.keras.layers.MaxPooling2D(2, 2),
        # tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        # tf.keras.layers.MaxPooling2D(2, 2),
        # tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        # tf.keras.layers.MaxPooling2D(2, 2),
        # tf.keras.layers.Flatten(),
        # tf.keras.layers.Dense(512, activation='relu'),
        # tf.keras.layers.Dense(1, activation='sigmoid')
    ])


def load_data(directory):
    return tf.keras.utils.image_dataset_from_directory(
        directory,
        labels='inferred',
        image_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
        batch_size=BATCH_SIZE,
    )


# Function to count images in a dataset
def count_images(dataset):
    return len(list(dataset))


# Load datasets
training_dataset = load_data('./dataset/test')
validation_dataset = load_data('./dataset/validation')

class_names = training_dataset.class_names

print("\nLoaded {} training batches per epoch".format(int(training_dataset.cardinality().numpy())))
print("Loaded {} validation batches per epoch".format(int(validation_dataset.cardinality().numpy())))

# Preprocessing: Standardize the data (rescale pixel values)
normalization_layer = tf.keras.layers.Rescaling(1. / 255)

training_dataset = (training_dataset.map(lambda x, y: (normalization_layer(x), y))
                    .cache()
                    .prefetch(buffer_size=tf.data.AUTOTUNE))
validation_dataset = (validation_dataset.map(lambda x, y: (normalization_layer(x), y))
                      .cache()
                      .prefetch(buffer_size=tf.data.AUTOTUNE))

model = create_model()

print("\n")
model.summary()

model.compile(optimizer=RMSprop(learning_rate=1e-4),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("\n")
history = model.fit(
    training_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS)

plot_stats.plot_loss_acc(history)

# 5. Save the model
model.save('model_64_64.keras')
print("Model saved")
