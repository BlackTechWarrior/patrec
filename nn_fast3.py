import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
import time

start_time = time.perf_counter()

def distanceBtnPixelsFaster(image1, image2):
    image1 = np.float32(image1)
    image2 = np.float32(image2)
    dif = image1 - image2
    sq = dif * dif
    total = sq.sum()
    return total ** 0.5

def nearestNeighbors(image1, training_images):
    smallest_dif = np.inf
    smallest_indx = -1
    
    for i in range(5000):
        suma = distanceBtnPixelsFaster(image1, training_images[i])
    
        if suma <= smallest_dif:
            smallest_dif = suma
            smallest_indx = i
    return smallest_indx

def evalTestRange(test_images):
    correct_cnt = 0
    incorrect_cnt = 0
    
    for i in range(1000):
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


correct, incorrect = evalTestRange(test_images)
print(f"correct: {correct}, incorrect: {incorrect}, accuracy: {correct/(correct + incorrect)}")

end_time = time.perf_counter()
print(f"elapsed time: {end_time - start_time}")

