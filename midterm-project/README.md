# Midterm Project ML Zoomcamp

## Predicting Employee Attrition

### Problem

Employee attrition is the process of employees leaving a company due to several reasons (for example: through resignation for personal reasons or retirement).
If a company wants to keep valuable employees the HR department should be made aware of a possible attrition based on several features.

This is what this prediction model is aimed to do, predicting the possibility that an employee might be set to leave.
T
The data set used is the *IBM HR Analytics Employee Attrition & Performance* and can be downloaded here: https://www.kaggle.com/pavansubhasht/ibm-hr-analytics-attrition-dataset

## Development

This repository contains the following files:

### Pipenv

- *Pipfile*, *Pipfile.lock*: The pipenv environment files.

To setup the pipenv environment on the local compouter simply run:

    pip --no-cache-dir install pipenv (in pipenv is not installed yet)

    pipenv install

The python code can then be run either in the pipenv shell:

    pipenv shell
    python train.py

or with the run command:

    pipenv run python train.py

### ML Code

- *WA_Fn-UseC_-HR-Employee-Attrition.csv*: the downloaded data set
- *notebook.ipynb*: the jupyter notebook with the data preparation, explorative data analysis and the tuning of several machine learning models.
- *train.py*: the python code exported out of the notebook that cleans, prepares the data and generates the machine learning models. The best performing model is saved as final model (attrition-model.bin).
- *predict.py*: the web service loading the saved machine learning model and offering a REST API to make predictions.
- *request.py*: a test script that selects randomly employee entries from the test data set (exported to 'attrition_test_data.csv') and sends them to the prediction web service.  

## Deployment

### Docker

Since I ran into version conflict problems, I created 2 docker images:

- one that creates the machine learning model
- one that offers the prediction web service

To build those docker containers, the following files are relevant:
- *Dockerfile-train*: the Dockerfile to build the 'train' container.
- *Dockerfile-predict*: the Dockerfile to build the 'prediction service' container.
- *docker_cmds.txt*: documentation about building and running those containers.

Be sure to run the 'train' container first, since this will create the machine learning model that is then used by the 'predition' container.
This model is saved on the Docker volume 'attrition-vol'.
The 'train' container is built as follows: 

    docker build -f Dockerfile-train -t attrition-train .

and can be started:

    docker run -it -v attrition-vol:/output attrition-train:latest

The  'prediction service' container: 

    docker build -f Dockerfile-predict -t attrition-predict .

and started:

    docker run -it -p 9696:9696 -v attrition-vol:/output attrition-predict:latest


### AWS Elastic Beanstalk


## Test

The model can be tested with the *request.py* file. The code goes in a loop and selects randomly one of the records in the test data set and sends it to the prediction service which returns a result that is printed on the console.

After each iteration one can stop the loop.

    pipenv run python request.py



