# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 22:06:33 2017

@author: vigou
"""

import _pickle
import gzip
import numpy as np


def load_data():
    f = gzip.open('mnist.pkl.gz', 'rb')
    tr_d, va_d, te_d = _pickle.load(f, encoding='iso-8859-1')
    f.close()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data =[]
    for i in range (0, len(training_inputs)):
        training_data.append((training_inputs[i], training_results[i]))
    #training_data = zip(training_inputs, training_results)
    #print(type(training_data[0]))
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    test_data = []
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    #test_data = zip(test_inputs, te_d[1])
    for i in range (0, len(test_inputs)):
        test_data.append((test_inputs[i], te_d[1][i]))
    return (training_data, validation_data, test_data)


def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e