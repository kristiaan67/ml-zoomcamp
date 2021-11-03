FROM python:3.8

ENV PYTHONUNBUFFERED=TRUE

RUN python -m pip install --upgrade pip

RUN pip --no-cache-dir install numpy scikit-learn flask gunicorn

WORKDIR /output
COPY ["attrition-model.bin", "./"]

WORKDIR /app
COPY ["predict.py", "./"]

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:9696", "predict:app"]