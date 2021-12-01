#!/usr/bin/env python
# coding: utf-8

import numpy as np
import tflite_runtime.interpreter as tflite

from keras_image_helper import create_preprocessor


preprocessor = create_preprocessor('xception', target_size=(150, 150))

interpreter = tflite.Interpreter(model_path='cats-dogs-v2.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
input_index = input_details[0]['index']

output_details = interpreter.get_output_details()
output_index = output_details[0]['index']


def predict(url):
	X = preprocessor.from_url(url)

	interpreter.set_tensor(input_index, X)
	interpreter.invoke()
	preds = interpreter.get_tensor(output_index)
	return preds[0]


def lambda_handler(event, context):
    url = event['url']
    preds = predict(url)
    print(preds)

    return {
        'probability': float(preds)
    }






