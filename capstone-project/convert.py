#!/usr/bin/env python
# coding: utf-8


## Imports

import tensorflow as tf
from tensorflow import keras

## Convert the model to TF lite format

model = keras.models.load_model('cats_and_dogs_v1_best.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with tf.io.gfile.GFile('cats_and_dogs_v1.tflite', 'wb') as f:
    f.write(tflite_model)


## Convert the model to 'saved_model' format

tf.saved_model.save(model, 'cats_and_dogs')