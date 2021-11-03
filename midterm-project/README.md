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

To setup the pipenv environment on the local computer run:

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

Note that there is also a *Dockerfile*, this is used for the AWS Elastic Beanstalk deployment.

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

The model can be tested with the *request.py* file. The code goes in a loop and selects randomly one of the records in the test data set and sends it to the prediction service which returns a result that is printed on the console.

Whether a local docker environment is used or the service on AWS Elastic Beanstalk is specified by the *--mode=* argument. It can take 2 values:

- **local**: local docker environment
- **aws**: AWS Elastic Beanstalk

After each iteration one can stop the loop.

So test the service on a local docker environment, one can:

    pipenv run python request.py --mode=local

or on AWS Elastic Beanstalk:

    pipenv run python request.py --mode=aws


Obviously one can also use other tools like *curl*. The endpoints are:

* Docker on local machine: http://localhost:9696/attrition_score
* AWS Elastic Beanstalk: http://attrition-predict-env.eba-smk4cc3d.us-west-2.elasticbeanstalk.com/attrition_score

Does somehow not work for me but just in case somebody wants to give it a try:

    curl -i -X POST http://localhost:9696/attrition_score -d "{'age': 48, 'businesstravel': 'Travel_Rarely', 'dailyrate': 969, 'department': 'Research & Development', 'distancefromhome': 2, 'education': 'College', 'educationfield': 'Technical Degree', 'environmentsatisfaction': 'very high', 'gender': 'Male', 'hourlyrate': 76, 'jobinvolvement': 'very high', 'joblevel': 1, 'jobrole': 'Laboratory Technician', 'jobsatisfaction': 'medium', 'maritalstatus': 'Single', 'monthlyincome': 2559, 'monthlyrate': 16620, 'numcompaniesworked': 5, 'overtime': 0, 'percentsalaryhike': 11, 'performancerating': 'excellent', 'relationshipsatisfaction': 'high', 'stockoptionlevel': 0, 'totalworkingyears': 7, 'trainingtimeslastyear': 4, 'worklifebalance': 'good', 'yearsatcompany': 1, 'yearsincurrentrole': 0, 'yearssincelastpromotion': 0, 'yearswithcurrmanager': 0}"

