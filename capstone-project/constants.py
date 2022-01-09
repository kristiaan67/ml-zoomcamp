#######################################
## Constants used in the other scripts
#######################################


DEBUG=False

LABELS = {
    0: 'cat',
    1: 'dog'
}

# All audio files will be resized to 7 seconds
AUDIO_DURATION = 7

# parameters to generate the Mel Spectograms
N_MELS=256
N_FFT=2048
HOP_LENGTH=512

# the dimensions of the Mel Spectogram images for the CNN
IMG_WIDTH=302
IMG_HEIGHT=256
