# Local Build Instructions:
# 1. From the repository root run the following
# $ docker build -f ./build/prodigy/Dockerfile .

# why cant this be ubuntu?
FROM python:3.7-bullseye

ENV PRODIGY_HOME=/home/prodigy
ENV PYTHONPATH "${PYTHONPATH}:${PRODIGY_HOME}"
WORKDIR /home/prodigy

RUN mkdir ./prodigy_utils
COPY ./prodigy_utils ./prodigy_utils/
COPY ./prodigy_json/prodigy.json ./
# assumes cloudbuild has downloaded wheel file from gcs
RUN pip3 install ./prodigy_utils/prodigy*.whl

COPY build/prodigy/requirements.txt ./
RUN pip3 install -r ./requirements.txt

RUN python -m nltk.downloader punkt

RUN cd ./prodigy_utils && python setup.py develop

ENTRYPOINT [ "/bin/bash", "-c" ]
EXPOSE 8080