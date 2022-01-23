#!/usr/bin/env python
# coding: utf-8


## Imports

import os

import numpy as np

import tflite_runtime.interpreter as tflite

import cv2

from constants import *
from functions import prepare_mel_spectogram_image

import urllib.request

from flask import Flask, flash, request, redirect


## Global Variables

MODEL_FILE = '/app/cats_and_dogs_v1.tflite'
AUDIOS_FOLDER = '/app/audios'
if not os.path.exists(AUDIOS_FOLDER):
    os.makedirs(AUDIOS_FOLDER)

IMAGES_FOLDER = '/app/images'
if not os.path.exists(AUDIOS_FOLDER):
    os.makedirs(AUDIOS_FOLDER)

app = Flask('Cats and Dogs App')
app.config['UPLOAD_FOLDER'] = AUDIOS_FOLDER

# Load ML model
interpreter = tflite.Interpreter(model_path=MODEL_FILE)
interpreter.allocate_tensors()
print(f"Loaded ML model: {MODEL_FILE}")    

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


def predict_animal_class_audio(audio_file):
    img_file_name = audio_file[audio_file.rfind("/")+1:]
    img_file_name = img_file_name.replace(".wav", ".png")
    img_file = f'{AUDIOS_FOLDER}/{img_file_name}'
    
    prepare_mel_spectogram_image(audio_file, img_file)
    return predict_animal_class_image(img_file)



## Routes

PAGE_HTML = '''
    <!doctype html>
    <title>Upload Cat or Dog Audio</title>
    <h1>Upload Cat or Dog Audio</h1>
    <form action="/dog_or_cat" method="post" enctype="multipart/form-data">
      <input type="file" name="file" accept="audio/wav"/>
      <br/><br/>
      <input type="submit" value="Process"/>
      <br/><br/><strong>Result:</strong> <span>@RESULT@</span>
    </form>
    '''

@app.route('/', methods=['GET'])
def index():
    return PAGE_HTML.replace('@RESULT@', '')

@app.route('/dog_or_cat', methods=['POST'])
def dog_or_cat():
    if 'file' not in request.files:
        flash('No file part')
        return redirect('/')

    audio_file = request.files['file']
    if audio_file.filename == '':
        flash('No selected file')
        return redirect('/')

    audio_filepath = os.path.join(app.config['UPLOAD_FOLDER'], audio_file.filename)
    audio_file.save(audio_filepath)
    
    result = predict_animal_class_audio(audio_filepath)
    print(f"Prediction result: {result}")
    return PAGE_HTML.replace('@RESULT@', result)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)
