""" main.py
    This script implements the gradient descent algorithm to 
    get the model for multivariate linear regression and apply logistic classification.

    Author:         Jonathan Nunez Rdz.
    Institution:    Universidad de Monterrey
    First Created:  Sat 18 April, 2020
    Email:          jonathan.nunez@udem.edu // jonathannunezr1@gmail.com
"""

def main():
    #   import standard libraries
    import numpy as np
    import pandas as pd

    #   import functions
    import utilityFunctions as uf
    import sys
    import time

    print("--- the program has started, please wait ---")
    start_time = time.time()

    #   define the name of the flie to read the data from
    csv = 'diabetes.csv'
    #   define the training data size, testing data size will be calculated accordingly
    training_size = 0.8
    #   define whether to plot the w values at the end of the program
    create_histogram = True

    #   display is required and is used for whether or not to print the training data set:
    #
    #   display = 1, will print to the console
    #   display = 0, will not print
    display = 0
    #   -------------------------------------------------------------------------------------------
    #   will_scale is required and it will define whether or not to implement feature 
    #   scalling for the testing data 
    # 
    #   will_scale = 1, will implement feature scalling 
    #   wii_scale = 0, will not
    will_scale_x = 1
    #   -------------------------------------------------------------------------------------------    # 
    #   uf.load_data will read the determined csv (must be local) with pandas and divide tha data to x_training 
    #   and y_training, the data will be divided in 80% to training and 20% to testing which will be used to 
    #   calculate the gradient of cost function.
    #
    #   it will also return the mean and deviation to use it to predict
    x_training, y_training, x_testing, y_testing, labels = uf.load_data(csv, display, will_scale_x, training_size)


    #   declare w to zeros, w will be the "weight" for each of the features in the data set, which will then be 
    #   calculated with the gradient descent method
    w = np.zeros([x_training.shape[1], 1])
    #   -------------------------------------------------------------------------------------------    
    #   display is optional and is used for whether or not to print w and L2_norm for each iteration in gradient descent
    #   WARNING!! --> printing will impact heavily the time it takes to calculate w, for low end machines leave it set 
    #                 equal to 0
    #
    #   display = 1, will print to the console
    #   display = 0, will not print
    display = 0
    #   -------------------------------------------------------------------------------------------
    #   stopping_criteria is used for stopping the iterations, the gradient of cost will calculate L2_norm each itaration
    #   and will stop when it reaches the stopping_criteria
    #
    #   learning_rate is used for lower the "force" of the gradient of cost, it is recommended to start at a low value,
    #   otherwise when it starts to "correct/update" the w values, there is a chance that it will go beyond the minimum and will
    #   try to correct again (there's also a chance that it will stay in an infinite loop)
    stopping_criteria = 0.01
    learning_rate = 0.0005
    #   -------------------------------------------------------------------------------------------
    #   calculate w values with the gradient descent method
    w, features_histogram = uf.gradient_descent_multivariate(x_training, y_training, w, stopping_criteria, learning_rate, display, create_histogram)


    #   make the predictions based on the test data (with feature scaling)
    y = uf.predict(w,x_testing)
    #   -------------------------------------------------------------------------------------------
    #   print_results takes as arguments the array of shape(n,1), the name of the rows, and the title to print before the values
    uf.print_results(w, 'w', 'w parameters')
    uf.print_results(y, 'y', 'outcome [prediction]', predictions=1)


    #   get_confusion_matrix will return a matrix with the true positives, true negatives, false positives, and false negatives
    #   -------------------------------------------------------------------------------------------
    confusion_matrix = uf.get_confussion_matrix(y,y_testing)
    #   print_performance_metrics will print the accuracy, precision, recall, specificity, and f1 score based on the confusion matrix
    #   -------------------------------------------------------------------------------------------
    uf.print_performance_metrics(confusion_matrix)

    print("--- the program has ended ---")
    print("--- %s seconds ---" % (time.time() - start_time))

    if create_histogram: uf.create_histogram(features_histogram, labels)


if __name__ == "__main__":
    main()