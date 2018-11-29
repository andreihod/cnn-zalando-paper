#!/usr/bin/env python
# -*- coding: utf-8 -*-
import keras, sys, time, datetime, os
import numpy as np
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.metrics import categorical_accuracy
from keras.preprocessing.image import ImageDataGenerator

img_rows    = 28
img_cols    = 28
epochs      = 500
batch_size  = 2048
ncategories = 10

today = datetime.datetime.now()
model_name = sys.argv[1] # pega por parâmetro
folder_name = './log/' + model_name + today.strftime('-%d-%m-%Y-%H-%M')
os.mkdir(folder_name)

output = model_name + '\n'

output_file = open(folder_name + '/output.txt', 'w')

tensorboard = keras.callbacks.TensorBoard( 
	log_dir=folder_name, histogram_freq=50, write_graph=False, write_images=False)

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

# Normalização
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Data augmentation
#train_datagen = ImageDataGenerator(zoom_range=0.2, horizontal_flip=True)

y_train = keras.utils.to_categorical(y_train, ncategories)
y_test = keras.utils.to_categorical(y_test, ncategories)

model = Sequential()

if model_name == 'cnn-dropout-1':
	model.add(Conv2D(filters=6, kernel_size=5, input_shape=(img_rows, img_cols, 1), activation='relu', kernel_initializer='he_normal'))
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
if model_name == 'cnn-dropout-2':
	model.add(Conv2D(filters=6, kernel_size=5, input_shape=(img_rows, img_cols, 1), activation='relu', kernel_initializer='he_normal'))
	model.add(Conv2D(filters=6, kernel_size=5, activation='relu', kernel_initializer='he_normal'))
	model.add(MaxPooling2D(strides=2))
	model.add(Dropout(0.20))
	model.add(Conv2D(filters=16, kernel_size=5, activation='relu', kernel_initializer='he_normal'))
	model.add(Conv2D(filters=16, kernel_size=5, activation='relu', kernel_initializer='he_normal'))
	model.add(MaxPooling2D(strides=2))
	model.add(Dropout(0.25))
	model.add(Conv2D(filters=32, kernel_size=5, activation='relu', kernel_initializer='he_normal'))
	model.add(Dropout(0.25))
	model.add(Flatten())
	model.add(Dense(120, activation = 'relu', kernel_initializer='he_normal'))
	model.add(Dropout(0.5))
	model.add(Dense(84, activation = 'relu', kernel_initializer='he_normal'))
	model.add(Dropout(0.5))
	model.add(Dense(ncategories, activation = 'softmax', kernel_initializer='he_normal'))
if model_name == 'cnn-simple-1':
	model.add(Conv2D(filters=6, kernel_size=5, input_shape=(img_rows, img_cols, 1), activation='relu', kernel_initializer='he_normal'))
	model.add(Conv2D(filters=12, kernel_size=5, activation='relu', kernel_initializer='he_normal'))
	model.add(MaxPooling2D(strides=2))
	model.add(Dropout(0.25))
	model.add(Flatten())
	model.add(Dense(90, activation = 'relu', kernel_initializer='he_normal'))
	model.add(Dropout(0.5))
	model.add(Dense(ncategories, activation = 'softmax', kernel_initializer='he_normal'))


model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

keras.utils.print_summary(model, print_fn=lambda x: output_file.write(x + '\n'))

start = time.time()

model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, 
							callbacks=[tensorboard], validation_data=(x_test, y_test))

keras.utils.plot_model(model, to_file=folder_name+'/model.png')

end = time.time()
output = output + 'Tempo: ' + str(end - start) + ' segundos\n'
output = output + 'Avaliando modelo...\n'

acc_train = model.evaluate(x_train, y_train, verbose=0)
output =  output + 'Train loss/accuracy: ' + str(acc_train[0]) + '/' + str(acc_train[1]*100) + '%\n'

acc_test = model.evaluate(x_test, y_test, verbose=0)
output = output + 'Test loss/accuracy: ' + str(acc_test[0]) + '/' + str(acc_test[1]*100) + '%\n'

print(output)

output_file.write(output)
output_file.close()