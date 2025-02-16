import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
import time

start_time = time.perf_counter()

def majority(neighbor_labels):
    counts = np.bincount(neighbor_labels)  #returns the frequency of each element in the array
    return np.argmax(counts)               #returns the max frequency in counts arr/list
        

def nearestNeighbors(image1, training_images, k):
    image1 = np.float32(image1)
    tr_beg = 0
    tr_end = 10000
    training_subset = np.float32(training_images[tr_beg:tr_end])
    diffs = training_subset - image1
    dists = np.linalg.norm(diffs.reshape(tr_end - tr_beg, -1), axis = 1)
    sorted_dists = np.argsort(dists) #sorting the distances from smallest to largest distance
    k_nearest = sorted_dists[:k]     #slicing sorted_dists to just the k-th nearest elems
    return k_nearest + tr_beg        #returning arr of size k-th nearest with any offest from tr_beg (if any)
    

def evalTestRange(test_images, k = 10):
    
    test_beg = 0 
    test_end = 1000
    
    predictions = [] #empty list to hold predictions in the for loop
    
    for test_img in test_images[:test_end]: 
        #initializing k_indices with the return closest k-elems to test_img
        k_indices = nearestNeighbors(test_img, training_images, k)
        
        #initializing neighbor_labels with training_labels[k_th indices]
        #both k_indices & neighbor_labels are lists 
        neighbor_labels = training_labels[k_indices]
        
        #here we want out "prediction" to be the majority of these neighbor labels
        prediction = majority(neighbor_labels)
        predictions.append(prediction)
    
    predictions = np.array(predictions)  #converting list to numpy array to match types with test_labels
    correct = np.sum(predictions == test_labels[test_beg:test_end])
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


