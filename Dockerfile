FROM ubuntu:latest

RUN apt-get update && \
    apt-get install -y \
    python \
    python-pip

ADD requirements.txt /opt/ml/requirements.txt

WORKDIR /opt/ml

RUN pip install -r requirements.txt

ADD . /opt/ml

CMD python predict.py
