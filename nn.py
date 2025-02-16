import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
import time

start_time = time.perf_counter()

#FUNCTION DEFINATIONS
#func that calcs and returns the difference btn any 2 image elements
#using euclidian distance formula
def distanceBtnPixels(image1, image2):
    image1 = np.float32(image1)
    image2 = np.float32(image2)
    suma = 0 #variable to hold the sum of distances between every image element(pixel)
    #nested for-loop because image is a 2d array. 28 by 28
    for row in range(28):
        for col in range(28):
            dist = (image1[row][col] - image2[row][col]) ** 2  #square dist
            suma = suma + dist 
    
    suma = suma ** 0.5  #sqrt suma
    return suma

#func that finds the nearest neighbor (smallest dif) image from an image in a specified range in the training images
#to a specified image in test_images
#func returns the indx in training_images that has the smallest difference
def nearestNeighbors(image1, training_images):
    smallest_dif = np.inf
    smallest_indx = -1
    
    for i in range(100):
        suma = distanceBtnPixels(image1, training_images[i])
        
        #if suma is less than or equal to smallest_dif, 
        #update smallest_dif to suma & smallest_indx to i
        if suma <= smallest_dif:
            smallest_dif = suma
            smallest_indx = i
    return smallest_indx

#top-level func that uses nearest neighbors to check a range of test_images to 
#a range of training images to 'guess' what number that test_image is
def evalTestRange(test_images):
    correct_cnt = 0
    incorrect_cnt = 0
    
    for i in range(100):
        tr_indx = nearestNeighbors(test_images[i], training_images)
        test_lbl = test_labels[i]
        training_lbl = training_labels[tr_indx] 
        
        if test_lbl == training_lbl:
            correct_cnt = correct_cnt + 1 
        else:
            incorrect_cnt = incorrect_cnt + 1 
            
    return correct_cnt, incorrect_cnt
      
#defining datasets
dataset = tf.keras.datasets.mnist
(training_set, test_set) = dataset.load_data()
(training_images, training_labels) = training_set
(test_images, test_labels) = test_set


#calling the evalTestRange function. Initializing two variables from return statements
#variables must be in order
#if you only needed one variable, replace what you dont need with ' _ ' e.g., correct, _ = evalTestRange(test_images)
#separate variables with a comma
correct, incorrect = evalTestRange(test_images)
print(f"correct: {correct}, incorrect: {incorrect}, accuracy: {correct/(correct + incorrect)}")

#calculating time to check improvements
end_time = time.perf_counter()
print(f"elapsed time: {end_time - start_time}")