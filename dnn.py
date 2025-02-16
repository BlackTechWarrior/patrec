import os 
import multiprocessing as mp

os.environ["TF_NUM_INTEROP_THREADS"] = str(mp.cpu_count())
os.environ["TF_NUM_INTRAOP_THREADS"] = str(mp.cpu_count())


import tensorflow as tf 
import numpy as np

# Constants
BATCH_SIZE = 5000
EPOCHS = 10
TRAINING_SIZE = 5000
TEST_SIZE = 1000

# Load dataset
dataset = tf.keras.datasets.mnist
(training_set, test_set) = dataset.load_data()
(training_images, training_labels) = training_set
(test_images, test_labels) = test_set

training_images = training_images[:TRAINING_SIZE]
training_labels = training_labels[:TRAINING_SIZE]
test_images = test_images[:TEST_SIZE]
test_labels = test_labels[:TEST_SIZE]

# Create data pipeline
train_ds = tf.data.Dataset.from_tensor_slices((training_images, training_labels))\
    .cache()\
    .map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))\
    .batch(BATCH_SIZE)\
    .prefetch(tf.data.AUTOTUNE)

test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))\
    .cache()\
    .map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))\
    .batch(BATCH_SIZE)

# Create MLP model
dnn_model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(512, activation='tanh'),
    tf.keras.layers.Dense(10, activation="softmax")
])

# Compile and train
dnn_model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

# Train the model
history = dnn_model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=test_ds
)

# Evaluate
test_loss, test_acc = dnn_model.evaluate(test_ds, verbose=0)
print('\nTest accuracy: %.2f%%' % (test_acc * 100))
