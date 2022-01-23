# Capstone Project ML Zoomcamp

## Audio Classification of Cats and Dogs

### Goal

The goal of the this project was to develop a classification model that processes audio files of barking dogs and
meowing cats and regonizes the animal. One can discuss about the usefulness of this model bit I was intrigued to
read that audio files are processed by the convolutional neural network (CNN) as images by converting them in so called **Mel Spectograms**.

The data set containing 277 audio files can be downloaded from Kaggle: https://www.kaggle.com/mmoreaux/audio-cats-and-dogs.

I followed some of the algortithms published in the following article series by Ketan Doshi:

https://towardsdatascience.com/audio-deep-learning-made-simple-part-1-state-of-the-art-techniques-da1d3dff2504

A good desription of Mel spectograms can be found here:

https://medium.com/analytics-vidhya/understanding-the-mel-spectrogram-fca2afa2ce53

## Development

This repository contains the following files:

### Pipenv

- *Pipfile*, *Pipfile.lock*: The pipenv environment files.

To setup the pipenv environment on the local computer run:

    pip --no-cache-dir install pipenv (if pipenv is not installed yet)

    pipenv install

The python code can then be run either in the pipenv shell:

    pipenv shell
    python train.py

or with the run command:

    pipenv run python train.py

Some common functions of the training and classification code is stored:


### ML Code

- *notebook.ipynb*: the jupyter notebook describing the data preparation (i.e. generation of the Mel Spectograms) and the tuning of the CNN
- *train.py*: the python code exported out of the notebook that generates the Mel Spectograms and the CNN. The best performing model is saved as final model: 'cats_and_dogs_v1_best.h5'.
- *convert.py*: the python code that converts the Tensor Flow model to TensorFlow Lite: 'cats_and_dogs_v1.tflite'
- *predict.py*: the web service loading the saved machine learning model and offering a (primitive) web frontend and REST API to upload .wav files and returning the classification.

## Deployment

### Docker

- *Dockerfile*: the Dockerfile to build the 'classification service' container.

Build the container: 

    docker build -f Dockerfile -t dogs_cats_audio_service:latest .

Start the container:

    docker run -it -p 9696:9696 dogs_cats_audio_service:latest

    
### AWS Elastic Beanstalk

To deploy the model on AWS Elastic Beanstalk, follow the following steps.

Install the AWS Client Interface:

    pipenv install awsebcli --dev

Start the pipenv shell:

    pipenv shell

Execute the following commands:

    eb init -p docker attrition-predict


To start/test the service locally:

    eb local run --port 9696

To finally deploy in the cloud:

    eb create attrition-predict-env


## Test

The model can be tested with the web frontend by uploading a .wav file. Some example wavfiles can be found in the 'test' directory of this repository.
The classification result is shown beneath the 'Process' button.

The endpoints are:

* Docker on local machine: http://localhost:9696/
* AWS Elastic Beanstalk: http://attrition-predict-env.eba-smk4cc3d.us-west-2.elasticbeanstalk.com/

