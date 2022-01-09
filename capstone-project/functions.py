######################################
# Functions used in the other scripts
######################################

## Imports

import numpy as np

import librosa
import skimage.io

from constants import *


## FUNCTIONS

def load_audio_file(file):
    audio, sample_rate = librosa.load(file)
    audio, _ = librosa.effects.trim(audio) # trim silent edges
    audio_length_sec = librosa.get_duration(y=audio, sr=sample_rate)
    if DEBUG:
        print(f"Loaded audio {file} of {audio_length_sec} seconds duration")
    return audio, sample_rate


def resize_audio(audio, sample_rate, duration):
    max_length = int(duration * sample_rate)
    while len(audio) < max_length:
        audio = np.concatenate([audio, audio])
        
    if len(audio) > max_length:    
        audio = audio[:max_length] # truncate the audio

    if DEBUG:
        print(f"Resized audio to {duration} seconds duration")
    return audio


def create_mel_spectogram(audio, sample_rate):
    sgram = librosa.stft(audio)
    mel_scale_sgram = librosa.feature.melspectrogram(S=librosa.magphase(sgram)[0], 
    												 sr=sample_rate,
                                                     n_mels=N_MELS,
                                                     n_fft=N_FFT, 
                                                     hop_length=HOP_LENGTH)
    mel_sgram = librosa.amplitude_to_db(mel_scale_sgram, ref=np.min)
    if DEBUG:
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
    audio = resize_audio(audio, sample_rate, AUDIO_DURATION)
    mel_sgram = create_mel_spectogram(audio, sample_rate)
    save_mel_spectogram_image(mel_sgram, output)



