# -*- coding: utf-8 -*-
import time
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))
    
def init_biases(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def model(X, w_h, w_o ,b_h, b_o, p_keep_input, p_keep_hidden):
    X = tf.nn.dropout(X, p_keep_input)
    h = tf.nn.relu(tf.add(tf.matmul(X, w_h),b_h))
    
    h = tf.nn.dropout(h, p_keep_hidden)
    #h2 = tf.nn.relu(tf.matmul(h, w_h2))

    #h2 = tf.nn.dropout(h2, p_keep_hidden)

    return tf.add(tf.matmul(h, w_o), b_o)

with tf.device("/gpu:0"):
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
    
    X = tf.placeholder("float", [None, 784])
    Y = tf.placeholder("float", [None, 10])
    
    w_h = init_weights([784, 800])
    #w_h2 = init_weights([625, 625])
    b_h = init_biases([800])
    w_o = init_weights([800, 10])
    b_o = init_biases([10])
    p_keep_input = tf.placeholder("float")
    p_keep_hidden = tf.placeholder("float")
    py_x = model(X, w_h, w_o, b_h, b_o, p_keep_input, p_keep_hidden)
    
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
    train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
    predict_op = tf.argmax(py_x, 1)

# Launch the graph in a session
t0 = time.time()
with tf.Session() as sess:
    # you need to initialize all variables
    tf.global_variables_initializer().run()

    for i in range(10):
        for start, end in zip(range(0, len(trX), 128), range(128, len(trX)+1, 128)):
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end],
                                          p_keep_input: 0.8, p_keep_hidden: 0.5})
        print(i, np.mean(np.argmax(teY, axis=1) ==
                         sess.run(predict_op, feed_dict={X: teX, 
                                                         p_keep_input: 1.0,
                                                         p_keep_hidden: 1.0})))
    alpha = w_h.eval()
    beta  = w_o.eval()
    delta = b_h.eval()
    gamma = b_o.eval()
t1 = time.time()-t0
print ("time needed")
print(t1)
