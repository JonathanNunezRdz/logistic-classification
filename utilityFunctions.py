""" utilityFunctions.py
    This script is used as a library, it contains several functions that are used for
    implemeting logistic classification and logistic regression to create a model for
    predicting.

    Author:         Jonathan Nunez Rdz.
                    Ezequiel
    Institution:    Universidad de Monterrey
    First Created:  Sat 18 April, 2020
    Email:          jonathan.nunez@udem.edu // jonathannunezr1@gmail.com
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import time
import itertools

"""load data from input csv by user"""
def load_data(file:str, will_scale_x:bool, training_size:float, display:bool):
    try:
        data = pd.read_csv(file)
    except:
        print_title("'./{}' file directory doesn't exist".format(file))
        exit(1)
    
    # split data into training and testing
    labels = data.columns.values
    training_data = data.sample(frac=training_size)
    testing_data = data.drop(training_data.index)

    # split training data into x's and y
    x_training = pd.DataFrame.to_numpy(training_data.iloc[:, 0:-1])
    y_training = pd.Series.to_numpy(training_data.iloc[:, -1]).reshape(x_training.shape[0],1)

    # split testing data into x's and y
    x_testing = pd.DataFrame.to_numpy(testing_data.iloc[: , 0:-1])
    y_testing = pd.Series.to_numpy(testing_data.iloc[:, -1]).reshape(x_testing.shape[0], 1)

    if display:
        print_title('Particular Explratory Data Analysis')
        print_subtitle('10 Training data samples')
        for i in range(10): print('\t'.join([np.format_float_positional(j,3) for j in x_training[i,:]]))
        print_subtitle('10 Testing data samples')
        for i in range(10): print('\t'.join([np.format_float_positional(j,3) for j in x_testing[i,:]]))
        print_statistics(x_training, 'Training')    
        print_statistics(x_testing, 'Testing')    

    # feature scalling for x if specified
    ones = np.ones([x_training.shape[0],1])
    if will_scale_x:
        # feature scalling for training
        x_training_scaled, mean, deviation = scale_x(x_training, 'training')
        x_training_scaled = np.concatenate((ones,x_training_scaled), axis=1)

        # feature scalling for testing
        x_testing_scaled = scale_x(x_testing, 'testing', mean=mean, deviation=deviation)

        # return the training data and testing data scaled
        return x_training_scaled, y_training, x_testing_scaled, y_testing, labels
    else:
        x_training = np.concatenate((ones,x_training), axis=1)
        # return the training data and testing data
        return x_training, y_training, x_testing, y_testing, labels

"""scale x_training to avoid conflicts"""
def scale_x(x, mode, **kwargs):
    if(mode == 'training'):
        x_scaled = np.zeros_like(x)
        mean = np.zeros([1, x_scaled.shape[1]])
        deviation = np.zeros([1, x_scaled.shape[1]])
        for i in range(x.shape[1]):
            mean[0,i] = x[:,i].mean()
            deviation[0,i] = x[:,i].std()
            for j in range(x.shape[0]): x_scaled[j,i] = (x[j,i] - mean[0,i]) / np.sqrt(deviation[0,i]**2+10**-8)
        return x_scaled,mean,deviation    
    if(mode == 'testing'):
        defined_mean = kwargs.get('mean')
        defined_deviation = kwargs.get('deviation')
        x_scaled = np.zeros_like(x)
        for i in range(x.shape[1]):
            for j in range(x.shape[0]): x_scaled[j,i] = (x[j,i] - defined_mean[0,i]) / np.sqrt(defined_deviation[0,i]**2+10**-8)
        return x_scaled

"""calculate w until the L2_norm reaches the stopping criteria"""
def gradient_descent_multivariate(x_training, y_training, w, stopping_criteria:float, learning_rate:float, display:bool, create_histogram:bool):
    print("--- the gradient descent has started, please wait ---")
    start_time = time.time()
    L2_norm = 100.0
    iterations = 0
    cost_function_histogram = np.zeros([0,w.shape[0]])
    if create_histogram: 
        for i in itertools.count():
            if L2_norm <= stopping_criteria:
                iterations = i
                break
            gradient_of_cost_function = compute_gradient_of_cost_function_multivariate(x_training, y_training, w)
            cost_function_histogram = np.append(cost_function_histogram, gradient_of_cost_function.T, axis=0)
            w = w - (learning_rate * gradient_of_cost_function)
            L2_norm = compute_L2_norm_multivariate(gradient_of_cost_function)
            if display: print('w: {}, L2: {}'.format(w, L2_norm))
    else:
        for i in itertools.count():
            if L2_norm <= stopping_criteria:
                iterations = i
                break            
            gradient_of_cost_function = compute_gradient_of_cost_function_multivariate(x_training, y_training, w)
            w = w - (learning_rate * gradient_of_cost_function)
            L2_norm = compute_L2_norm_multivariate(gradient_of_cost_function)
            if display: print('w: {}, L2: {}'.format(w, L2_norm))        
    seconds = time.time() - start_time
    print("--- the gradient descent has ended, see resuls ---")
    print("--- {} seconds ---".format(seconds))
    print("--- {} iterations ---".format(iterations))
    if create_histogram: return w, cost_function_histogram
    return w, 0

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
    return 1.0/(1 + np.exp(-np.dot(w.T, x.T)))

"""calculate the L2_norm based with the gradient of cost function"""
def compute_L2_norm_multivariate(gradient_of_cost_function):
    return np.sqrt(np.sum(gradient_of_cost_function**2))

"""print results"""
def print_results(results, name, title, **kwargs):
    print_title(title)
    if kwargs.get('predictions'):
        with_diabetes = 0
        without_diabetes = 0
        for i in range(results.shape[0]):
            print("{}[{}]: {}\t=> {}".format(name,i,results[i,0],results[i,1]))
            if results[i,1] == 1: with_diabetes += 1
            else: without_diabetes += 1
        print("with diabetes\t\t=> {}".format(with_diabetes))
        print("without diabetes\t=> {}".format(without_diabetes))
    else: 
        for i in range(results.shape[0]): print("{}[{}]: {}".format(name,i,results[i,0]))

"""predict the y values for a x_test input"""
def predict(w, x):
    y = np.zeros([x.shape[0],1])
    for i in range(x.shape[0]):   
        y[i] += w[0]      
        for j in range(1,w.shape[0]):
            y[i] += w[j]*x[i,j-1]
    new_y = np.zeros([x.shape[0],2])
    for i in range(new_y.shape[0]):
            new_y[i,0] = 1/(1 + math.exp(-y[i,0]))
            if new_y[i,0] >= 0.5: new_y[i,1] = 1
            else: new_y[i,1] = 0
    return new_y

"""calculate the confusion matrix with the given predicted class and the actual class"""
def get_confussion_matrix(predicted_class, actual_class):
    confusion_matrix = np.zeros([2,2])
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    for i in range(predicted_class.shape[0]):
        if(predicted_class[i,1] == 1 and actual_class[i,0] == 1): TP += 1
        elif (predicted_class[i,1] == 0 and actual_class[i,0] == 0): TN += 1
        elif (predicted_class[i,1] == 1 and actual_class[i,0] == 0): FP += 1
        else: FN += 1

    confusion_matrix[0,0] = TP
    confusion_matrix[1,1] = TN
    confusion_matrix[0,1] = FP
    confusion_matrix[1,0] = FN

    return confusion_matrix

"""print to the console the performance metrics"""
def print_performance_metrics(confusion_matrix):
    TP = confusion_matrix[0,0]
    TN = confusion_matrix[1,1]
    FP = confusion_matrix[0,1]
    FN = confusion_matrix[1,0]

    accuracy = (TP + TN)/(TP + TN + FP + FN)
    precision = TP /(TP + FP)
    recall = TP/(TP + FN)
    specificity = TN/(TN + FP)
    f1_score = 2*(precision * recall)/(precision + recall)

    print_title("confusion matrix")
    print('true positives\t\t=> {}'.format(TP))
    print('true negatives\t\t=> {}'.format(TN))
    print('false positives\t\t=> {}'.format(FP))
    print('false negatives\t\t=> {}'.format(FN))
    print_title("scores")
    print('accuracy\t\t=> {}'.format(accuracy))
    print('precision\t\t=> {}'.format(precision))
    print('recall\t\t\t=> {}'.format(recall))
    print('specificity\t\t=> {}'.format(specificity))
    print('f1 score\t\t=> {}'.format(f1_score))

"""create a plot with the features and their values across the interation number"""
def create_histogram(features_histogram, labels):
    plt.subplot(111)
    plt.xlabel('# of iteration')
    plt.ylabel("Cost")
    plt.title('Cost of function histogram')

    x_plot = np.arange(features_histogram.shape[0])
    for i in range(features_histogram.shape[1]):
        plt.plot(x_plot, np.array(features_histogram[:,i]), label=labels[i])
    
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()

def print_title(title:str):
    """
        print a title with a format
    """
    print("-"*40)
    print(title)
    print("-"*40)

def print_subtitle(subtitle:str, **kwargs):
    """
        print a subtitle with a format
    """
    print("-"*20)
    if kwargs.get('extra') != None: print(subtitle, kwargs.get('extra'))
    else: print(subtitle)
    print("-"*20)

def print_statistics(data, title):
    """
        print a subtitle with a format
    """
    print_title('Statistics - {}'.format(title))
    print_subtitle('Min (per feature)    =>', extra=[data[:,i].min() for i in range(data.shape[1])])
    print_subtitle('Max (per feature)    =>', extra=[data[:,i].max() for i in range(data.shape[1])])
    print_subtitle('Mean (per feature)   =>', extra=[float(np.format_float_positional(data[:,i].mean(), 3)) for i in range(data.shape[1])])
    print_subtitle('Median (per feature) =>', extra=[float(np.format_float_positional(np.median(data[:,i]))) for i in range(data.shape[1])])
