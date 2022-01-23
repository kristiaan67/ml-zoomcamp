FROM python:3.8

ENV PYTHONUNBUFFERED=TRUE

RUN python -m pip install --upgrade pip
RUN pip --no-cache-dir install pipenv
RUN apt-get update
RUN apt-get install -y python3-opencv
RUN apt-get install -y libsndfile1

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --deploy --system && rm -rf /root/.cache
RUN pip install --extra-index-url https://google-coral.github.io/py-repo/ tflite_runtime 

COPY ["cats_and_dogs_v1.tflite", "constants.py", "functions.py", "predict.py", "./"]

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:9696", "predict:app"]