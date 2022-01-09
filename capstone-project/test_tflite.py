#!/usr/bin/env python
# coding: utf-8


## Imports

import os
import glob

import numpy as np

import tflite_runtime.interpreter as tflite

import cv2

from constants import *
from functions import prepare_mel_spectogram_image


## Global Variables

# the CNN model
model_file = './cats_and_dogs_v1.tflite'


## Functions


def predict_animal_class_image(interpreter, img_file):
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


def predict_animal_class_audio(interpreter, audio_file):
    img_file_name = audio_file[audio_file.rindex("/")+1:].replace(".wav", ".png")
    img_file = f'./tmp/{img_file_name}'
    prepare_mel_spectogram_image(audio_file, img_file)
    return predict_animal_class_image(interpreter, img_file)


def test_predictions_audios(interpreter, level, exp_label):
    num = 0
    num_correct = 0
    for audio_file in sorted(glob.glob(f'./dataset/cats_dogs/{level}/{exp_label}/*.wav')):
        pred = predict_animal_class_audio(interpreter, audio_file)
        if DEBUG:
            print(f'Prediction {audio_file}: {exp_label} = {pred}')
        num = num + 1
        if (pred == exp_label):
            num_correct = num_correct + 1
    return num, num_correct



## Run Application
if __name__ == '__main__':

    interpreter = tflite.Interpreter(model_path=model_file)
    interpreter.allocate_tensors()

    num_cats, num_correct_cats = test_predictions_audios(interpreter, 'test', 'cat')
    print(f'Cats: {num_correct_cats}/{num_cats} audio predictions correct: {float(num_correct_cats/num_cats)}')

    num_dogs, num_correct_dogs = test_predictions_audios(interpreter, 'test', 'dog')
    print(f'Dogs: {num_correct_dogs}/{num_dogs} audio predictions correct: {float(num_correct_dogs/num_dogs)}')

    print(f'Total: {(num_correct_cats + num_correct_dogs)}/{(num_cats + num_dogs)} predictions correct: {float((num_correct_cats + num_correct_dogs)/(num_cats + num_dogs))}')


