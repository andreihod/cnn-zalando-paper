#!/usr/bin/env python
# -*- coding: utf-8 -*-
import keras
import numpy as np
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.metrics import categorical_accuracy
import tensorflow as tf
import time

img_rows = 28
img_cols = 28
epochs = 1
batch_size = 2048
ncategories = 10

model_name = 'cnn-dropout-1'

tensorboard = keras.callbacks.TensorBoard( 
	log_dir='./tmp/cnn-dropout-1', histogram_freq=50, write_graph=False, write_images=True)

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

# Normalização
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

y_train = keras.utils.to_categorical(y_train, ncategories)
y_test = keras.utils.to_categorical(y_test, ncategories)

model = Sequential()

model.add(Conv2D(filters=6, kernel_size=5, input_shape=(img_rows, img_cols, 1), 
				 activation='relu', kernel_initializer='he_normal'))
model.add(MaxPooling2D(strides=2))
model.add(Dropout(0.25))

model.add(Conv2D(filters=16, kernel_size=5, activation='relu', kernel_initializer='he_normal'))
model.add(MaxPooling2D(strides=2))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(120, activation = 'relu', kernel_initializer='he_normal'))
model.add(Dropout(0.5))

model.add(Dense(84, activation = 'relu', kernel_initializer='he_normal'))
model.add(Dropout(0.5))

model.add(Dense(ncategories, activation = 'softmax', kernel_initializer='he_normal'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

start = time.time()

model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, 
							callbacks=[tensorboard], validation_data=(x_test, y_test))

end = time.time()
print('Tempo: ' + str(end - start) + ' segundos')
print('Avaliando modelo...')

acc_train = model.evaluate(x_train, y_train, verbose=0)
print('Train loss/accuracy: ' + str(acc_train[0]) + '/' + str(acc_train[1]*100) + '%')

acc_test = model.evaluate(x_test, y_test, verbose=0)
print('Test loss/accuracy: ' + str(acc_test[0]) + '/' + str(acc_test[1]*100) + '%')