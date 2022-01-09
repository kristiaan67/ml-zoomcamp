FROM python:3.8

ENV PYTHONUNBUFFERED=TRUE

RUN python -m pip install --upgrade pip
RUN pip --no-cache-dir install pipenv

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --deploy --system && rm -rf /root/.cache

COPY ["cats_and_dogs_v1_best.h5", "constants.py", "functions.py", "predict.py", "./"]

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:9696", "predict:app"]