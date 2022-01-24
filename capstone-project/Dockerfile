FROM python:3.8

ENV PYTHONUNBUFFERED=TRUE

RUN apt-get update
RUN apt-get install -y python3-opencv
RUN apt-get install -y libsndfile1

WORKDIR /app

RUN python -m pip install --upgrade pip

RUN pip install numpy librosa scikit-image opencv-python flask gunicorn
RUN pip install --extra-index-url https://google-coral.github.io/py-repo/ tflite_runtime 

COPY ["cats_and_dogs_v1.tflite", "constants.py", "functions.py", "predict.py", "./"]

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:9696", "predict:app"]
