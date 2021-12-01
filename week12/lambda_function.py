#!/usr/bin/env python
# coding: utf-8

import numpy as np

import tflite_runtime.interpreter as tflite

from io import BytesIO
from urllib import request

from PIL import Image


interpreter = tflite.Interpreter(model_path='cats-dogs-v2.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
input_index = input_details[0]['index']

output_details = interpreter.get_output_details()
output_index = output_details[0]['index']


def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img


def preprocess_input(x):
    x /= 255
    return x

def predict(url):
	img = download_image(url)
	img = prepare_image(img, target_size=(150, 150))
	
	x = np.array(img, dtype='float32')
	X = preprocess_input(np.array([x]))
	
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






