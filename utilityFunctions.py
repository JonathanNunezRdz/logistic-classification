"""custom user functions"""

import pandas as pd
import numpy as np
import math
import sys

"""load data from input csv by user"""
def load_data(file, display, will_scale_x, training_size):
    try:
        data = pd.read_csv(file)
    except:
        print("-"*40)
        print("'./{}' file directory doesn't exist".format(file))
        print("-"*40)
        exit(1)
    
    # split data into training and testing
    training_data = data.sample(frac=training_size)
    testing_data = data.drop(training_data.index)

    if(display == 1):
        print('-'*40)
        print("training data set")
        print('-'*40)
        print(training_data)
        print('-'*40)
        print("testing data set")
        print('-'*40)
        print(testing_data)

    # split training data into x's and y
    x_training = pd.DataFrame.to_numpy(training_data.iloc[:, 0:-1])
    y_training = pd.Series.to_numpy(training_data.iloc[:, -1]).reshape(x_training.shape[0],1)

    # split testing data into x's and y
    x_testing = pd.DataFrame.to_numpy(testing_data.iloc[: , 0:-1])
    y_testing = pd.Series.to_numpy(testing_data.iloc[:, -1]).reshape(x_testing.shape[0], 1)

    # feature scalling for x if specified
    ones = np.ones([x_training.shape[0],1])
    if(will_scale_x == 1):
        # feature scalling for training
        x_training_scaled, mean, deviation = scale_x(x_training, 'training')
        x_training_scaled = np.concatenate((ones,x_training_scaled), axis=1)

        # feature scalling for testing
        x_testing_scaled = scale_x(x_testing, 'testing', mean=mean, deviation=deviation)       

        # display option
        if(display == 1):
            print('-'*40)
            print("training features scaled")
            print('-'*40)
            for i in range(x_training_scaled.shape[0]):
                print(x_training_scaled[i])
            print('-'*40)
            print("testing features scaled")
            print('-'*40)
            for i in range(x_testing_scaled.shape[0]):
                print(x_testing_scaled[i])

        # return the training data and testing data scaled
        return x_training_scaled, y_training, x_testing_scaled, y_testing
    else:
        x_training = np.concatenate((ones,x_training), axis=1)
        # return the training data and testing data
        return x_training, y_training, x_testing, y_testing

"""scale x_training to avoid conflicts"""
def scale_x(x, mode, **kwargs):
    if(mode == 'training'):
        x_scaled = np.zeros_like(x)
        mean = np.zeros(shape = (x_scaled.shape[1], 1))
        deviation = np.zeros(shape = (x_scaled.shape[1], 1))

        for i in range(x.shape[1]):

            col = np.zeros_like(x)

            for j in range(x.shape[0]):
                col[j] = x[j][i]
            
            mean[i] = col.mean()
            deviation[i] = col.std()

            for j in range(x.shape[0]):
                x_scaled[j][i] = (x[j][i] - col.mean()) / col.std()

        return x_scaled,mean,deviation
    
    if(mode == 'testing'):
        defined_mean = kwargs.get('mean')
        defined_deviation = kwargs.get('deviation')
        x_scaled = np.zeros_like(x)

        for i in range(x.shape[1]):
            for j in range(x.shape[0]):
                x_scaled[j][i] = (x[j][i] - defined_mean[i]) / defined_deviation[i]
        
        return x_scaled

"""calculate w until the L2_norm reaches the stopping criteria"""
def gradient_descent_multivariate(x_training, y_training, w, stopping_criteria, learning_rate, display):
    iteration = 0
    
    L2_norm = 100.0
    while L2_norm > stopping_criteria:
        gradient_of_cost_function = compute_gradient_of_cost_function_multivariate(x_training, y_training, w)
        w = w - (learning_rate * gradient_of_cost_function)
        L2_norm = compute_L2_norm_multivariate(gradient_of_cost_function)
        iteration += 1

        if(display == 1):
            print('w: {}, L2: {}'.format(w, L2_norm))

    print("--- {} L2_norm ---".format(L2_norm))
    print("--- {} iterations ---".format(iteration))
    return w

"""compute the cost of the gradient"""
def compute_gradient_of_cost_function_multivariate(x,y,w):

    N = x.shape[0]

    hypothesis_function = eval_hypothesis_function_multivariate(w, x)

    residual = np.subtract(hypothesis_function, y.T)

    gradient_of_cost_function = ((residual * x.T).sum(axis=1)/N)

    gradient_of_cost_function = np.reshape(gradient_of_cost_function, (gradient_of_cost_function.shape[0], 1))

    return gradient_of_cost_function

"""get hypothesis function with w and x"""
def eval_hypothesis_function_multivariate(w, x):
    return np.matmul(w.T, x.T)

"""calculate the L2_norm based with the gradient of cost function"""
def compute_L2_norm_multivariate(gradient_of_cost_function):
    return np.sqrt(np.sum(gradient_of_cost_function**2))

"""print results"""
def print_results(results, name, title, **kwargs):
    print('-'*40)
    print(title)
    print('-'*40)
    if(kwargs.get('predictions') == 1):
        with_diabetes = 0
        without_diabetes = 0
        for i in range(results.shape[0]):
            print("{}[{}]: {} => {}".format(name,i,results[i][0],results[i][1]))
            if(results[i][1] == 1):
                with_diabetes += 1
            else:
                without_diabetes += 1
        print("with diabetes\t\t=> {}".format(with_diabetes))
        print("without diabetes\t=> {}".format(without_diabetes))
    else:
        for i in range(results.shape[0]):
            print("{}[{}]: {}".format(name,i,results[i][0]))

"""predict the y values for a x_test input"""
def predict(x, w):    
    y = np.zeros([x.shape[0],1])

    for i in range(x.shape[0]):   
        y[i] += w[0]      
        for j in range(1,w.shape[0]):
            y[i] += w[j]*x[i][j-1]

    new_y = np.zeros([x.shape[0],2])
    for i in range(new_y.shape[0]):
            new_y[i][0] = 1.0 / (1.0 + math.exp(-y[i]))
            if(new_y[i][0] >= 0.5):
                new_y[i][1] = 1
            else:
                new_y[i][1] = 0

    return new_y

"""calculate the confusion matrix with the given predicted class and the actual class"""
def calculate_confussion_matrix(predicted_class, actual_class):
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    for i in range(predicted_class.shape[0]):
        if(predicted_class[i][1] == 1 and actual_class[i][0] == 1): TP += 1
        elif (predicted_class[i][1] == 0 and actual_class[i][0] == 0): TN += 1
        elif (predicted_class[i][1] == 1 and actual_class[i][0] == 0): FP += 1
        else: FN += 1

    accuracy = (TP + TN)/(TP + TN + FP + FN)
    precision = TP /(TP + FP)
    recall = TP/(TP + FN)
    specificity = TN/(TN + FP)

    print('-'*40)
    print('confussion matrix')
    print('-'*40)
    print('true positives\t\t=> {}'.format(TP))
    print('true negatives\t\t=> {}'.format(TN))
    print('false positives\t\t=> {}'.format(FP))
    print('false negatives\t\t=> {}'.format(FN))
    print('-'*40)
    print('scores')
    print('-'*40)
    print('accuracy\t\t=> {}'.format(accuracy))
    print('precision\t\t=> {}'.format(precision))
    print('recall\t\t\t=> {}'.format(recall))
    print('specificity\t\t=> {}'.format(specificity))