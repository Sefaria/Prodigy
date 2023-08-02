# Local Build Instructions:
# 1. From the repository root run the following
# $ docker build -f ./build/prodigy/Dockerfile .

# why cant this be ubuntu?
FROM gcr.io/production-deployment/base-web:latest

ENV PRODIGY_HOME=/home/prodigy
ENV PYTHONPATH "${PYTHONPATH}:${PRODIGY_HOME}"
WORKDIR /home/prodigy

RUN mkdir ./prodigy_utils
COPY ./prodigy_utils/* ./prodigy_utils/
# assumes cloudbuild has downloaded wheel file from gcs
COPY ./prodigy*.whl .
RUN pip3 install ./prodigy*.whl

COPY build/prodigy/requirements.txt ./
RUN pip3 install -r ./requirements.txt

RUN python -m nltk.downloader punkt

RUN cd ./prodigy_utils && python setup.py develop

ENTRYPOINT [ "/bin/bash", "-c" ]
EXPOSE 8080