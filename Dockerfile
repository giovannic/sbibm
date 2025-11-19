FROM python:3.9

ADD . /opt
WORKDIR /opt

RUN pip install -e ".[dev]"
