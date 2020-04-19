"""custom user functions"""

import pandas as pd
import numpy as np
import time

"""load data from input csv by user"""
def load_data_train(file, display):
    try:
        training_data = pd.read_csv(file)
    except:
        print("-"*40)
        print("'./{}' file directory doesn't exist".format(file))
        print("-"*40)
        exit(1)

    if(display == 1):
        print('-'*40)
        print("training data set")
        print('-'*40)
        print(training_data)
    
    n_samples = training_data.shape
    x_training = pd.DataFrame.to_numpy(training_data.iloc[:, 0:-1])
    y_training = pd.Series.to_numpy(training_data.iloc[:, -1]).reshape(n_samples[0],1)

    x_scaled,mean,deviation = scale_x(x_training, 'training')

    if(display == 1):
        print('-'*40)
        print("features scaled")
        print('-'*40)
        for i in range(x_scaled.shape[0]):
            print(x_scaled[i])

    ones = np.ones([x_scaled.shape[0],1])

    x_scaled = np.concatenate((ones,x_scaled), axis=1)

    if(display == 1):
        print('-'*40)
        print("features scaled with ones")
        print('-'*40)
        for i in range(x_scaled.shape[0]):
            print(x_scaled[i])

    return x_scaled, y_training, mean, deviation

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
    print("--- the program has started, please wait ---")
    start_time = time.time()
    iteration = 0
    
    L2_norm = 100.0
    stop = 0
    while L2_norm > stopping_criteria:
        gradient_of_cost_function = compute_gradient_of_cost_function_multivariate(x_training, y_training, w)
        w = w - (learning_rate * gradient_of_cost_function)
        L2_norm = compute_L2_norm_multivariate(gradient_of_cost_function)
        iteration += 1

        if(display == 1):
            print('w: {}, L2: {}'.format(w, L2_norm))

    print("--- {} iterations ---".format(iteration))
    print("--- %s seconds ---" % (time.time() - start_time))
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
def print_results(results, name, title):
    print('-'*40)
    print(title)
    print('-'*40)
    for i in range(results.shape[0]):
        print("{}[{}]: {}".format(name,i,results[i][0]))

"""load data from input csv by user"""
def load_data_test(file, mean, deviation, display):
    try:
        testing_data = pd.read_csv(file)
    except:
        print("-"*40)
        print("'./{}' file directory doesn't exist".format(file))
        print("-"*40)
        exit(1)
    

    if(display == 1):
        print('-'*40)
        print("testing data set")
        print('-'*40)
        print(testing_data)
    
    x_test = pd.DataFrame.to_numpy(testing_data.iloc[:, :])
    x_scaled = scale_x(x_test, 'testing', mean=mean, deviation=deviation)

    if(display == 1):
        print('-'*40)
        print("x test data set [unscaled]")
        print('-'*40)
        for i in range(x_test.shape[0]):
            print(x_test[i])
        print('-'*40)
        print("x test data set [scaled]")
        print('-'*40)
        for i in range(x_scaled.shape[0]):
            print(x_scaled[i])

    return x_scaled

"""predict the y values for a x_test input"""
def predict(x, w):    
    y = np.zeros([x.shape[0],1])

    for i in range(x.shape[0]):
        y[i] = w[0] + w[1]*x[i][0] + w[2]*x[i][1] + w[3]*x[i][2] + w[4]*x[i][3]

    return y
        