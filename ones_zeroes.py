import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
import time

#defining datasets
dataset = tf.keras.datasets.mnist
(training_set, test_set) = dataset.load_data()
(training_images, training_labels) = training_set
(test_images, test_labels) = test_set

#dummy code to check for ones and zeroes
#loop through the first 10000 images in training set
#variables to calc accuracy
correct_cnt = 0
incorrect_cnt = 0

start_time = time.perf_counter()
for i in range(len(training_images)):
    example_img = training_images[i]
    label = training_labels[i]
    
    #loop to check for just ones & zeroes  
    #ignores other values
    if label == 1 or label == 0:
        #whitespace check. if greater than 30 assume is a 1, elsewise 0
        if (example_img.mean() < 30 and label == 1) or (example_img.mean() > 30 and label == 0):
            correct_cnt = correct_cnt + 1
        else:
            incorrect_cnt = incorrect_cnt + 1

print(f"correct: {correct_cnt}, incorrect: {incorrect_cnt}, accuracy: {correct_cnt/(correct_cnt + incorrect_cnt)}")

end_time = time.perf_counter()
print(f"execution time: {end_time - start_time}")

#counter to show inefficiency for this method when data labels look similar
# using 4 and 9
four_sum = 0
four_cnt = 0
nine_sum = 0
nine_cnt = 0
for i in range(60000):
    example_img = training_images[i]
    label = training_labels[i]
    
    if label == 4:
        four_sum = four_sum + example_img.mean()
        four_cnt = four_cnt + 1
    elif label == 9:
        nine_sum = nine_sum + example_img.mean()
        nine_cnt = nine_cnt + 1

print(f"four_mean: {four_sum / four_cnt}, nine_mean: {nine_sum / nine_cnt}")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    