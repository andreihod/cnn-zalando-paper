#!/usr/bin/env python
# -*- coding: utf-8 -*-
import keras
import numpy as np
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.metrics import categorical_accuracy
import tensorflow as tf

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

tensorboard = keras.callbacks.TensorBoard( 
	log_dir='./tmp', histogram_freq=10, write_graph=True, 
	write_grads=True, write_images=True)

img_size = 28

x_train = x_train.reshape(x_train.shape[0], img_size**2)
x_test = x_test.reshape(x_test.shape[0], img_size**2)

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

model = Sequential()

model.add(Dense(units=120, activation='relu', input_dim=img_size**2))
model.add(Dense(units=80, activation='relu'))
model.add(Dense(units=10, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=700, batch_size=1014, callbacks=[tensorboard], validation_data=(x_test, y_test))

acc_train = model.evaluate(x_train, y_train)
print('Train loss/accuracy: ' + str(acc_train[0]) + '/' + str(acc_train[1]*100) + '%')

acc_test = model.evaluate(x_test, y_test)
print('Test loss/accuracy: ' + str(acc_test[0]) + '/' + str(acc_test[1]*100) + '%')