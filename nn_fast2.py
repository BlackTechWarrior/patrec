import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
import time

start_time = time.perf_counter()

#numpy's library can directly compute the difference between 2 matrices using element
#wise subtraction(making it signif faster). this is done by: image1 - image2
#.ravel() takes the 28x28 2d array and flattens it to a 1d array of 784 elems
#np.linalg.norm is basically a python rep of matrix norm formula. This formula is used to
#calc the distance between 2 matrices
def distanceBtnPixels(image1, image2):
    image1 = np.float32(image1)
    image2 = np.float32(image2)
    return np.linalg.norm((image1 - image2).ravel())


def nearestNeighbors(image1, training_images):
        #Calculate distances for all training images at once. 
        #distances is basically an array formed with values that constitute the difference btn a 
        #specific test_image to training_images in a set range. Here we use array slicing. 
        #e.g., training_images[:1000] = training_images(0, 1000) = training_images[range(1000)] = training_images(1000)
    tr_range = 5000
    distances = np.array([distanceBtnPixels(image1, train_img) for train_img in training_images[:tr_range]])
    # Return index of minimum distance
    return np.argmin(distances)


#using the previous eval function doesn't change exe time by much. This function is only cleaner as it 
#combines calculations in singular lines, making it a taaaad bit faster 
def evalTestRange(test_images):
    
    test_range = 1000
    
    # Get prediction labels for 0 to range test images in one array
    predictions = np.array([nearestNeighbors(test_img, training_images) for test_img in test_images[:test_range]])
    
    # Compare predictions with actual labels in one loop and sum the times prediction = actual test labels
    correct = np.sum(training_labels[predictions] == test_labels[:test_range])
    incorrect = test_range - correct
    
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
