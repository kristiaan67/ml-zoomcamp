#!/usr/bin/env python
# coding: utf-8


## Imports

import os
import glob

import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from constants import *
from functions import prepare_mel_spectogram_image


## Global Variables

# the CNN model
model_file = './cats_and_dogs_v1_best.h5'


## Functions

def predict_animal_class_image(model, img_file):
    img = load_img(img_file, target_size=(IMG_HEIGHT, IMG_WIDTH))
    X = np.array([img_to_array(img)])  # Convert single image to a batch.
    y_pred = model.predict(X)
    return LABELS[int(y_pred[0].argmax())]


def predict_animal_class_audio(model, audio_file):
    img_file_name = audio_file[audio_file.rindex("/")+1:].replace(".wav", ".png")
    img_file = f'./tmp/{img_file_name}'
    prepare_mel_spectogram_image(audio_file, img_file)
    return predict_animal_class_image(model, img_file)


def test_predictions_audios(model, level, exp_label):
    num = 0
    num_correct = 0
    for audio_file in sorted(glob.glob(f'./dataset/cats_dogs/{level}/{exp_label}/*.wav')):
        pred = predict_animal_class_audio(model, audio_file)
        if DEBUG:
            print(f'Prediction {audio_file}: {exp_label} = {pred}')
        num = num + 1
        if (pred == exp_label):
            num_correct = num_correct + 1
    return num, num_correct


def test_predictions_images(model, level, exp_label):
    num = 0
    num_correct = 0
    for img_file in sorted(glob.glob(f'./dataset/images/{level}/{exp_label}/*.png')):
        pred = predict_animal_class_image(model, img_file)
        if DEBUG:
            print(f'Prediction {img_file}: {exp_label} = {pred}')
        num = num + 1
        if (pred == exp_label):
            num_correct = num_correct + 1
    return num, num_correct


## Run Application
if __name__ == '__main__':

    if not os.path.exists("./tmp"):
        os.makedirs("./tmp")


    ## Load ML model
    model = keras.models.load_model(model_file)
    print(f"Machine Learning Model loaded from '{model_file}'")

    num_cats, num_correct_cats = test_predictions_audios(model, 'test', 'cat')
    print(f'Cats: {num_correct_cats}/{num_cats} audio predictions correct: {float(num_correct_cats/num_cats)}')

    num_dogs, num_correct_dogs = test_predictions_audios(model, 'test', 'dog')
    print(f'Dogs: {num_correct_dogs}/{num_dogs} audio predictions correct: {float(num_correct_dogs/num_dogs)}')

    print(f'Total: {(num_correct_cats + num_correct_dogs)}/{(num_cats + num_dogs)} predictions correct: {float((num_correct_cats + num_correct_dogs)/(num_cats + num_dogs))}')

    #predict_animal_class_audio('./dataset/cats_dogs/test/cat/cat_14.wav')
    #predict_animal_class_audio('./dataset/cats_dogs/train/dog/dog_barking_0.wav')
    #predict_animal_class_image('./dataset/images/train/dog/orig_dog_barking_0.png')
