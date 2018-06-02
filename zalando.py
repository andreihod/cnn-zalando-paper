#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# constantes
# tamanho das imagens no dataset
IMG_SIZE       = 28
LEARNING_RATE  = 0.01
NUM_EPOCHS     = 1000
MINIBATCH_SIZE = 512

data = input_data.read_data_sets('data/fashion', one_hot=True)

X = tf.placeholder(tf.float32, [IMG_SIZE**2, None], name="X") # 28*28 = 784
Y = tf.placeholder(tf.float32, [10, None], name="Y")          # 10 labels

num_minibatches = int(data.train.images.shape[0]/MINIBATCH_SIZE)

# 784 -> 30 -> 15 -> 10
def initialize_parameters():
    W1 = tf.get_variable("W1", [30, IMG_SIZE**2], initializer = tf.contrib.layers.xavier_initializer())
    b1 = tf.get_variable("b1", [30, 1], initializer = tf.zeros_initializer())
    W2 = tf.get_variable("W2", [15, 30], initializer = tf.contrib.layers.xavier_initializer())
    b2 = tf.get_variable("b2", [15, 1], initializer = tf.zeros_initializer())
    W3 = tf.get_variable("W3", [10, 15], initializer = tf.contrib.layers.xavier_initializer())
    b3 = tf.get_variable("b3", [10, 1], initializer = tf.zeros_initializer())

    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3}
    

def forward_propagation(X, parameters):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    Z1 = tf.add(tf.matmul(W1, X), b1)
    A1 = tf.nn.relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)
    A2 = tf.nn.relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)
    return Z3

def compute_cost(Z3, Y):
    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = logits, labels = labels))


print('Iniciando...')

# Forward propagation para computar Z3
parameters = initialize_parameters()
Z3 = forward_propagation(X, parameters)
cost = compute_cost(Z3, Y)

optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:

    sess.run(init)

    for epoch in range(NUM_EPOCHS):

        epoch_cost = 0

        for _ in range(num_minibatches):

            batch_xs, batch_ys = data.train.next_batch(MINIBATCH_SIZE)
            # necessário transpor pois está no formato [instances, features]
            batch_xs = np.transpose(batch_xs)
            batch_ys = np.transpose(batch_ys)
            _ , minibatch_cost = sess.run([optimizer, cost], feed_dict={X:batch_xs, Y:batch_ys})
            epoch_cost += minibatch_cost / num_minibatches

        if epoch % 100 == 0:
            print ("Custo no epoch %i: %f" % (epoch, epoch_cost))

    parameters = sess.run(parameters)
    print ("Treinamento concluído!")

    correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    print ("Acuracia do Treino:", accuracy.eval({X: np.transpose(data.train.images), Y: np.transpose(data.train.labels)}))
    print ("Acuracia do Teste:", accuracy.eval({X: np.transpose(data.test.images), Y: np.transpose(data.test.labels)}))