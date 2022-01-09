#!/usr/bin/env python
# coding: utf-8


## Imports

import numpy as np

import tflite_runtime.interpreter as tflite

import cv2

from constants import *
from functions import prepare_mel_spectogram_image

import urllib.request

from flask import Flask, request, jsonify


## Global Variables

app = Flask('Cats and Dogs App')

model_file = '/app/cats_and_dogs_v1.tflite'

# Load ML model
interpreter = tflite.Interpreter(model_path=model_file)
interpreter.allocate_tensors()
    

## Functions

def predict_animal_class_image(img_file):
    img = cv2.imread(img_file)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))

    #input_shape = input_details[0]['shape']
    X = np.array(np.expand_dims(img, 0), dtype=np.float32)
    
    input_details = interpreter.get_input_details()
    input_index = input_details[0]['index']

    interpreter.set_tensor(input_index, X)
    interpreter.invoke()

    output_details = interpreter.get_output_details()
    output_index = output_details[0]['index']
    y_pred = interpreter.get_tensor(output_index)

    return LABELS[int(y_pred[0].argmax())]


def predict_animal_class_audio(audio_url):
    img_file_name = audio_url[audio_url.rfind("/")+1:]
    if img_file_name.rfind("?") > 0:
        img_file_name = img_file_name[:img_file_name.rfind("?")]
    img_file_name = img_file_name.replace(".wav", ".png")
    img_file = f'/app/tmp/{img_file_name}'
    
    urllib.request.urlretrieve(audio_url, img_file)
    prepare_mel_spectogram_image(audio_file, img_file)
    return predict_animal_class_image(img_file)



@app.route('/dog_or_cat', methods=['POST'])
def dog_or_cat():
    data = request.get_json()
    audio_url = data['url']
    print(f"Processing sound file: {audio_url}")
    result = predict_animal_class_audio(audio_url)
    print(f"Prediction result: {result}")
    return jsonify(res)


if __name__ == '__main__':ß
    app.run(debug=True, host='0.0.0.0', port=9696)
