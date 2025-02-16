import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt

#defining datasets
dataset = tf.keras.datasets.mnist
(training_set, test_set) = dataset.load_data()
(training_images, training_labels) = training_set
(test_images, test_labels) = test_set

#getting to print and plot specific values by index
tr_index = 1
example_img = training_images[tr_index]
label = training_labels[tr_index]
plt.imshow(example_img, cmap = 'gray')

#extra array print funcs 
print(f"dtype: {example_img.dtype}")
print(f"shape: {example_img.shape}")
print(f"class label: {label}")
