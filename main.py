""" main.py
    This script implements the gradient descent algorithm to 
    get the model for multivariate linear regression.

    Author:         Jonathan Nunez Rdz.
    Institution:    Universidad de Monterrey
    First Created:  Fri 27 March, 2020
    Email:          jonathan.nunez@udem.edu // jonathannunezr1@gmail.com
"""

def main():
    #   import standard libraries
    import numpy as np
    import pandas as pd

    #   import functions
    import utilityFunctions as uf


    #   display is optional and is used for whether or not to print the training data set:
    #
    #   display = 1, will print to the console
    #   display = 0, will not print
    #   -------------------------------------------------------------------------------------------
    #   uf.load_data will read the determined csv (must be local) with pandas and divide tha data to x_training 
    #   (with feature scaling) and y_training, which will be used to calculate the gradient of cost function.
    #
    #   it will also return the mean and deviation to use it to predict
    display = 0

    x_training,y_training,mean,deviation = uf.load_data_train('training-data-multivariate.csv', display)


    #   declare w to zeros, w will be the "weight" for each of the features in the data set, which will then be 
    #   calculated with the gradient descent method
    #   -------------------------------------------------------------------------------------------    
    #   display is optional and is used for whether or not to print w and L2_norm for each iteration in gradient descent
    #   WARNING!! --> printing will impact heavily the time it takes to calculate w, for low end machines leave it set 
    #                 equal to 0
    #
    #   display = 1, will print to the console
    #   display = 0, will not print
    #   -------------------------------------------------------------------------------------------
    #   stopping_criteria is used for stopping the iterations, the gradient of cost will calculate L2_norm each itaration
    #   and will stop when it reaches the stopping_criteria
    #
    #   learning_rate is used for lower the "force" of the gradient of cost, it is recommended to start at a low value,
    #   otherwise when it starts to "correct/update" the w values, there is a chance that it will go beyond the minimum and will
    #   try to correct again (there's also a chance that it will stay in an infinite loop)
    #   -------------------------------------------------------------------------------------------
    #   calculate w values with the gradient descent method
    w = np.zeros([x_training.shape[1], 1])

    display = 0    

    stopping_criteria = 0.01
    learning_rate = 0.0005

    w = uf.gradient_descent_multivariate(x_training, y_training, w, stopping_criteria, learning_rate, display)


    #   print_results takes as arguments the array of shape(n,1), the name of the rows, and the title to print before the values
    uf.print_results(w, 'w', 'w parameters')


    #   display is optional and is used for whether or not to print the testing data set:
    #
    #   display = 1, will print to the console
    #   display = 0, will not print
    #   -------------------------------------------------------------------------------------------
    #   load_data_test will read the determined csv (must be local) with pandas, this csv must only contain test data, which will
    #   make the predictions with
    #
    #   this time, mean and deviation are sent as parameters to scale the test data the same way the training data was scaled
    display = 0
    x_test = uf.load_data_test('testing-data-multivariate.csv', mean, deviation, display)


    #   make the predictions based on the test data (with feature scaling)
    #   -------------------------------------------------------------------------------------------
    #   print_results takes as arguments the array of shape(n,1), the name of the rows, and the title to print before the values
    y = uf.predict(x_test, w)
    uf.print_results(y, 'price', 'last-mile cost [prediction]')


if __name__ == "__main__":
    main()