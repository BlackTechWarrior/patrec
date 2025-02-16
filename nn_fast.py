import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
import time

start_time = time.perf_counter()

'''
    the function below basically does distanceBtnPixels & nearestNeighbors:
        - diffs is an array of matrix subtractions from the range of training_images and 
            set test_image. eg., diffs is a 1000by28by28 3d array
        - in dists: 
            diffs is flattened along it's 1st axis aka 1000. 
            diffs is now a 1000 by 784 2d array
            -1 : is basically a generalized to auto calculate new dimension size instead of hardcoding diffs.reshape(1000, 784)
        -similarly, argmin returns the index of the minimum distance
'''      
def nearestNeighbors(image1, training_images):
    image1 = np.float32(image1)
    tr_beg = 2000
    tr_end = 3000
    training_subset = np.float32(training_images[tr_beg:tr_end])
    diffs = training_subset - image1
    dists = np.linalg.norm(diffs.reshape(tr_end - tr_beg, -1), axis = 1)
    smallest_indx = np.argmin(dists)
    return smallest_indx + tr_beg


def evalTestRange(test_images):
    
    test_beg = 0 #if we are slicing from 0 anyways we can ignore hardcoding it in range
    test_end = 1000
    
    # Get prediction labels for  test images(test_beg,test_end) in one array
    predictions = np.array([nearestNeighbors(test_img, training_images) for test_img in test_images[:test_end]])
    
    # Compare predictions with actual labels
    correct = np.sum(training_labels[predictions] == test_labels[test_beg:test_end])
    incorrect = test_end - correct
    
    return correct, incorrect

# Load dataset
dataset = tf.keras.datasets.mnist
(training_set, test_set) = dataset.load_data()
(training_images, training_labels) = training_set
(test_images, test_labels) = test_set

# Get accuracy
correct, incorrect = evalTestRange(test_images)
print(f"correct: {correct}, incorrect: {incorrect}, accuracy: {correct/(correct + incorrect)}")

end_time = time.perf_counter()
print(f"elapsed time: {end_time - start_time}")

