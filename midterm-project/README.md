# Midterm Project ML Zoomcamp

## Predicting Employee Attrition

Employee attrition is the process of employees leaving a company for example, through resignation for personal reasons or retirement.
This prediction model tries to predict whether a employee might leave the company or not.

The data set used is the *IBM HR Analytics Employee Attrition & Performance* and can be downloaded here: https://www.kaggle.com/pavansubhasht/ibm-hr-analytics-attrition-dataset

## Development

This repository contains the following files:

- *WA_Fn-UseC_-HR-Employee-Attrition.csv*: the downloaded data set
- *notebook.ipynb*: the jupyter notebook with the data preparation, explorative data analysis and the tuning of several machine learning models.
- *train.py*: the python code generated out of the notebook that cleans, prepares the data and generates the machine learning models. The best performing model is saved as final model (attrition-model.bin).
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


The 'train' container is built as follows: 

docker build -f Dockerfile-train -t attrition-train .

and can be started:

docker run -it -v attrition-vol:/output attrition-train:latest

The  'prediction service' container: 

docker build -f Dockerfile-predict -t attrition-predict .

and started:

docker run -it -p 9696:9696 -v attrition-vol:/output attrition-predict:latest
