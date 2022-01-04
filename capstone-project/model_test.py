#!/usr/bin/env python
# coding: utf-8


## Imports

import os
import glob

import numpy as np  # linear algebra

import librosa
import skimage.io

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array

import matplotlib.pyplot as plt

## Constants

# Log levels
debug = False

labels = {
    0: 'cat',
    1: 'dog'
}

# the CNN model
model_file = './cats_and_dogs_v1_best.h5'

# duration (seconds) of the audio file
audio_duration = 7

# parameters to generate the Mel Spectograms
n_fft=2048
hop_length=512
n_mels=256

# the dimensions of the Mel Spectogram images for the CNN
img_width=302
img_height=256


def load_audio_file(file):
    audio, sample_rate = librosa.load(file)
    audio, _ = librosa.effects.trim(audio) # trim silent edges
    audio_length_sec = librosa.get_duration(y=audio, sr=sample_rate)
    if debug:
        print(f"Loaded audio {file} of {audio_length_sec} seconds duration")
    return audio, sample_rate


def resize_audio(audio, sample_rate, duration):
    max_length = int(duration * sample_rate)
    while len(audio) < max_length:
        audio = np.concatenate([audio, audio])
        
    if len(audio) > max_length:    
        audio = audio[:max_length] # truncate the audio

    if debug:
        print(f"Resized audio to {duration} seconds duration")
    return audio


def create_mel_spectogram(audio, sample_rate, n_mels, n_fft, hop_length):
    sgram = librosa.stft(audio)
    mel_scale_sgram = librosa.feature.melspectrogram(S=librosa.magphase(sgram)[0], 
                                                     sr=sample_rate,
                                                     n_mels=n_mels,
                                                     n_fft=n_fft, 
                                                     hop_length=hop_length)
    mel_sgram = librosa.amplitude_to_db(mel_scale_sgram, ref=np.min)
    if debug:
        print(f"Created Mel Spectogram")
    return mel_sgram


def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled


def save_mel_spectogram_image(mel_sgram, output):
    img = scale_minmax(mel_sgram, 0, 255).astype(np.uint8)
    img = np.flip(img, axis=0) # put low frequencies at the bottom in image
    img = 255 - img # invert. make black==more energy

    # save as PNG
    skimage.io.imsave(output, img)


def prepare_mel_spectogram_image(input, output):
    # generate the Mel Spectogram Image
    audio, sample_rate = load_audio_file(input)
    audio = resize_audio(audio, sample_rate, audio_duration)
    mel_sgram = create_mel_spectogram(audio, sample_rate, n_mels, n_fft, hop_length)
    save_mel_spectogram_image(mel_sgram, output)


def predict_animal_class_image(img_file):
    img = load_img(img_file, target_size=(img_height,img_width))
    X = np.array([img_to_array(img)])  # Convert single image to a batch.
    y_pred = model.predict(X)
    return labels[int(y_pred[0].argmax())]


def predict_animal_class_audio(audio_file):
    img_file_name = audio_file[audio_file.rindex("/")+1:].replace(".wav", ".png")
    img_file = f'./tmp/{img_file_name}'
    prepare_mel_spectogram_image(audio_file, img_file)
    return predict_animal_class_image(img_file)


def test_predictions_audios(level, exp_label):
    num = 0
    num_correct = 0
    for audio_file in sorted(glob.glob(f'./dataset/cats_dogs/{level}/{exp_label}/*.wav')):
        pred = predict_animal_class_audio(audio_file)
        if debug:
            print(f'Prediction {audio_file}: {exp_label} = {pred}')
        num = num + 1
        if (pred == exp_label):
            num_correct = num_correct + 1
    return num, num_correct


def test_predictions_images(level, exp_label):
    num = 0
    num_correct = 0
    for img_file in sorted(glob.glob(f'./dataset/images/{level}/{exp_label}/*.png')):
        pred = predict_animal_class_image(img_file)
        if debug:
            print(f'Prediction {img_file}: {exp_label} = {pred}')
        num = num + 1
        if (pred == exp_label):
            num_correct = num_correct + 1
    return num, num_correct


## Start Application

if not os.path.exists("./tmp"):
    os.makedirs("./tmp")


## Load ML model
model = keras.models.load_model(model_file)
print(f"Machine Learning Model loaded from '{model_file}'")

num_cats, num_correct_cats = test_predictions_audios('test', 'cat')
print(f'Cats: {num_correct_cats}/{num_cats} audio predictions correct: {float(num_correct_cats/num_cats)}')

num_dogs, num_correct_dogs = test_predictions_audios('test', 'dog')
print(f'Dogs: {num_correct_dogs}/{num_dogs} audio predictions correct: {float(num_correct_dogs/num_dogs)}')

print(f'Total: {(num_correct_cats + num_correct_dogs)}/{(num_cats + num_dogs)} predictions correct: {float((num_correct_cats + num_correct_dogs)/(num_cats + num_dogs))}')

#predict_animal_class_audio('./dataset/cats_dogs/test/cat/cat_14.wav')
predict_animal_class_audio('./dataset/cats_dogs/train/dog/dog_barking_0.wav')
predict_animal_class_image('./dataset/images/train/dog/orig_dog_barking_0.png')
