#!/usr/bin/env python
# coding: utf-8


## Imports

import os
import glob
import zipfile

import pandas as pd
import numpy as np

import librosa
import skimage.io

import tensorflow as tf
from tensorflow import keras

from constants import *

## Global Variables

ROOT_DIR  = './dataset/cats_dogs'
DATA_FILE = './dataset.zip'
IMG_DIR   = './dataset/images'

SEED = 2022

## Set up

np.random.seed(SEED)

### Extract the dataset zip file
if not os.path.exists(ROOT_DIR):
    with zipfile.ZipFile(DATA_FILE, 'r') as zip_ref:
        zip_ref.extractall("./")


## Load the Audio Files

def load_audio_file(type, label, file):
    audio, sample_rate = librosa.load(file)
    audio, _ = librosa.effects.trim(audio) # trim silent edges
    audio_length_sec = librosa.get_duration(y=audio, sr=sample_rate)
    return type, label, file, audio, sample_rate, audio_length_sec


def load_audio_files():
    audios = []
    for type in ['train', 'test']:
        for label in ['cat', 'dog']:
            for file in glob.glob(f'{ROOT_DIR}/{type}/{label}/*.wav'):
                audios.append(load_audio_file(type, label, file))
    return pd.DataFrame(audios, columns=['type', 'label', 'file', 'audio', 'sample_rate', 'duration'])


df = load_audio_files()
df['variant'] = 'orig'
print("Number of 'cat' training audios: {}".format(len(df[(df['type'] == 'train') & (df['label'] == 'cat')])))
print("Number of 'dog' training audios: {}".format(len(df[(df['type'] == 'train') & (df['label'] == 'dog')])))
print("Number of 'cat' test audios: {}".format(len(df[(df['type'] == 'test') & (df['label'] == 'cat')])))
print("Number of 'dog' test audios: {}".format(len(df[(df['type'] == 'test') & (df['label'] == 'dog')])))

## Prepare the Audio Files

### Resize audios to the same length

def resize_audio(audio, sample_rate):
    max_length = int(AUDIO_DURATION * sample_rate)
    while len(audio) < max_length:
        audio = np.concatenate([audio, audio])
        
    if len(audio) > max_length:    
        audio = audio[:max_length] # truncate the audio

    return audio


df['resized_audio'] = df.apply(lambda x: resize_audio(x['audio'], x['sample_rate']), axis=1)
df['resized_duration'] = AUDIO_DURATION


### Augmenting the Audio Files

def noise_injection(audio, sample_rate, noise_factor=.01):
    noise = np.random.randn(len(audio))
    ni = audio + noise * noise_factor
    return ni


def time_shift(audio, sample_rate, shift_rate=.5):
    shift = int(sample_rate * shift_rate) # lenght of sample rate is 1 second
    ts = np.roll(audio, shift)
    return ts


def time_stretch(audio, sample_rate, stretch_rate=.25):
    audio_length = len(audio)
    stretch = 1 + np.random.randint(-100*stretch_rate, 100*stretch_rate) / 100
    ts = librosa.effects.time_stretch(audio, stretch)
    return resize_audio(ts, sample_rate)


def pitch_shift(audio_signals, sample_rate, max_steps=4):
    n_steps = np.random.randint(-max_steps, max_steps)
    ps = librosa.effects.pitch_shift(audio_signals, sample_rate, n_steps=n_steps)
    return ps


def augment_audio(audio):
    aug = np.random.randint(0, 4)
    augmented_audio = audio
    if aug == 0:
        augmented_audio = noise_injection(audio['resized_audio'], audio['sample_rate'])
        return augmented_audio, 'noise_injection'
    if aug == 1:
        augmented_audio = time_shift(audio['resized_audio'], audio['sample_rate'])
        return augmented_audio, 'time_shift'
    if aug == 2:
        augmented_audio = time_stretch(audio['resized_audio'], audio['sample_rate'])
        return augmented_audio, 'time_stretch'
    if aug == 3:
        augmented_audio = pitch_shift(audio['resized_audio'], audio['sample_rate'])
        return augmented_audio, 'pitch_shift'
    return augmented_audio, 'none'


def augment_audio_df(df):
    res = df.copy()
    for index, audio in df[df['type'] == 'train'].iterrows():
        augmented_audio, variant = augment_audio(audio)
        new_audio = audio.copy()
        new_audio['resized_audio'] = augmented_audio;
        new_audio['variant'] = audio['variant'] + '_' + variant;
        res = res.append(new_audio)

    return res


aug_df = augment_audio_df(augment_audio_df(df)) # augment twice
print("Number of augmented 'cat' training audios: {}".format(len(aug_df[(aug_df['type'] == 'train') & (aug_df['label'] == 'cat')])))
print("Number of augmented 'dog' training audios: {}".format(len(aug_df[(aug_df['type'] == 'train') & (aug_df['label'] == 'dog')])))


### Generating Mel Spectograms

def create_mel_spectogram(audio, sample_rate):
    sgram = librosa.stft(audio)
    mel_scale_sgram = librosa.feature.melspectrogram(S=librosa.magphase(sgram)[0], 
                                                     sr=sample_rate,
                                                     n_mels=N_MELS,
                                                     n_fft=N_FFT, 
                                                     hop_length=HOP_LENGTH)
    return librosa.amplitude_to_db(mel_scale_sgram, ref=np.min)


def create_mel_spectogram_df(df):
    res = df.copy()
    res['mel_sgram'] = res.apply(lambda x: create_mel_spectogram(x['resized_audio'], x['sample_rate']), axis=1)
    return res


final_df = create_mel_spectogram_df(aug_df)


### Saving the Audio Images


def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled

def save_mel_spectogram_image(mel_sgram, file):
    img = scale_minmax(mel_sgram, 0, 255).astype(np.uint8)
    img = np.flip(img, axis=0) # put low frequencies at the bottom in image
    img = 255 - img # invert. make black==more energy

    # save as PNG
    skimage.io.imsave(file, img)
    
def save_mel_spectogram_images_df(df):
    for type in ['train', 'test']:
        for label in ['cat', 'dog']:
            dirName = f"{IMG_DIR}/{type}/{label}"
            if not os.path.exists(dirName):
                os.makedirs(dirName)
            
    for index, audio in df.iterrows():
        audio_file_name = audio['file'][audio['file'].rindex("/")+1:].replace(".wav", ".png")
        file = f"{IMG_DIR}/{audio['type']}/{audio['label']}/{audio['variant']}_{audio_file_name}"
        save_mel_spectogram_image(audio['mel_sgram'], file)


save_mel_spectogram_images_df(final_df)
print("Saved Audio Images")


## ML Model

### Create train, validation and test datasets

def load_mel_spectogram_images():
    train_directory = './dataset/images/train'
    test_directory = './dataset/images/test'
    
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_directory, labels='inferred', label_mode='int', image_size=(IMG_HEIGHT, IMG_WIDTH), 
        validation_split=0.2, subset='training', seed=SEED)

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_directory, labels='inferred', label_mode='int', image_size=(IMG_HEIGHT, IMG_WIDTH), 
        validation_split=0.2, subset='validation', seed=SEED)

    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_directory, labels='inferred', label_mode='int', image_size=(IMG_HEIGHT, IMG_WIDTH), 
        validation_split=None, subset=None)
    
    return train_ds, val_ds, test_ds


train_ds, val_ds, test_ds = load_mel_spectogram_images()

### Create and tune the ML model

def create_model(num_classes, learning_rate, dropout_rate=0):
    model = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.Rescaling(1./255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes)
        ])
        
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer, loss_fn, ['accuracy'])

    return model

    
def find_best_model(models):
    best_model = 0
    best_accuracy = 0
    for idx, model in enumerate(models):
        loss, accuracy = model.evaluate(test_ds)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = idx

    return best_model


num_classes = len(train_ds.class_names)
num_epochs = 15


#### Tune learning rate

learning_rates = [0.001, 0.01, 0.1]

models = []
for learning_rate in learning_rates:
    model = create_model(num_classes, learning_rate)
    models.append(model)
    print(f"\nFitting with learning rate {learning_rate}")
    model.fit(train_ds, epochs=num_epochs, validation_data=val_ds)

best_model = find_best_model(models)
best_learning_rate = learning_rates[best_model]
print(f'The best learning rate: {best_learning_rate}')


#### Tune dropout

dropout_rates = [0.1, 0.2, 0.3, 0.4, 0.5]

models = []
for dropout_rate in dropout_rates:
    model = create_model(num_classes, best_learning_rate, dropout_rate=dropout_rate)
    models.append(model)
    print(f"\nFitting with dropout rate {dropout_rate}")
    model.fit(train_ds, epochs=num_epochs, validation_data=val_ds)

best_model = find_best_model(models)
best_dropout_rate = dropout_rates[best_model]
print(f'The best dropout rate: {best_dropout_rate}')


#### Generating the best model

num_epochs = 30
model = create_model(num_classes, best_learning_rate, dropout_rate=best_dropout_rate)
checkpoint = keras.callbacks.ModelCheckpoint(
    "cats_and_dogs_v1_{epoch:02d}_{val_accuracy:.3f}.h5",
    save_best_only=True,
    monitor="val_accuracy"
)
model.fit(train_ds, epochs=num_epochs, validation_data=val_ds, callbacks=[checkpoint])


print("Done.")
