import os 
import multiprocessing as mp

os.environ["TF_NUM_INTEROP_THREADS"] = str(mp.cpu_count()/2)
os.environ["TF_NUM_INTRAOP_THREADS"] = str(mp.cpu_count()/2)

import tensorflow as tf 
import numpy as np


# Constants
BATCH_SIZE = 128 
EPOCHS = 10

# Load dataset
dataset = tf.keras.datasets.mnist
(training_set, test_set) = dataset.load_data()
(training_images, training_labels) = training_set
(test_images, test_labels) = test_set

TRAINING_SIZE = len(training_images)
TEST_SIZE = len(test_images)

training_images = training_images[:TRAINING_SIZE]
training_labels = training_labels[:TRAINING_SIZE]
test_images = test_images[:TEST_SIZE]
test_labels = test_labels[:TEST_SIZE]

# Create data pipeline
train_ds = tf.data.Dataset.from_tensor_slices((training_images, training_labels))\
    .cache()\
    .map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))\
    .map(lambda x, y: (tf.expand_dims(x, -1), y))\
    .batch(BATCH_SIZE)\
    .prefetch(tf.data.AUTOTUNE)

test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))\
    .cache()\
    .map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))\
    .map(lambda x, y: (tf.expand_dims(x, -1), y))\
    .batch(BATCH_SIZE)

# Create CNN model
cnn_model = tf.keras.Sequential([
    tf.keras.Input(shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation="softmax"),
])

# Compile and train
cnn_model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

# Train the model
history = cnn_model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=test_ds
)

# Evaluate
test_loss, test_acc = cnn_model.evaluate(test_ds, verbose=0)
print('\nTest accuracy: %.2f%%' % (test_acc * 100))