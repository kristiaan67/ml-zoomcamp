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
COPY ["./dataset/cats_dogs/test/cat/cat_14.wav", "./dataset/cats_dogs/test/cat/cat_137.wav", "./"] 
COPY ["./dataset/cats_dogs/test/dog/dog_barking_112.wav", "./dataset/cats_dogs/test/dog/dog_barking_43.wav", "./"]

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:9696", "predict:app"]
