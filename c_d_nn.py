import tensorflow as tf
import numpy as np

# consts
EPOCHS = 10
TRAINING_SIZE = 5000
TEST_SIZE = 1000

# Load dataset
dataset = tf.keras.datasets.mnist
(training_set, test_set) = dataset.load_data()
(training_images, training_labels) = training_set
(test_images, test_labels) = test_set

#training data prep
training_images = training_images[:TRAINING_SIZE].astype('float32') / 255.0
training_labels = training_labels[:TRAINING_SIZE]
test_images = test_images[:TEST_SIZE].astype('float32') / 255.0
test_labels = test_labels[:TEST_SIZE]

# cnn channel dimensions
cnn_train_images = np.expand_dims(training_images, -1)
cnn_test_images = np.expand_dims(test_images, -1)

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

# Compile and train CNN
cnn_model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

# Train CNN model
cnn_history = cnn_model.fit(
    cnn_train_images,
    training_labels,
    epochs=EPOCHS,
    validation_data=(cnn_test_images, test_labels) #this is extra, u can remove
)

# Evaluate CNN
cnn_test_loss, cnn_test_acc = cnn_model.evaluate(cnn_test_images, test_labels, verbose=0)
print('\nCNN Test accuracy: %.2f%%' % (cnn_test_acc * 100))

# Create DNN model
dnn_model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(512, activation='tanh'),
    tf.keras.layers.Dense(10, activation="softmax")
])

# Compile and train DNN
dnn_model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

# Train DNN model
dnn_history = dnn_model.fit(
    training_images,
    training_labels,
    epochs=EPOCHS,
    validation_data=(test_images, test_labels)
)

# Evaluate DNN
dnn_test_loss, dnn_test_acc = dnn_model.evaluate(test_images, test_labels, verbose=0)
print('\nDNN Test accuracy: %.2f%%' % (dnn_test_acc * 100))
