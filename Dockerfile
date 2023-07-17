FROM chronis10/teaching-base:amd64 as amd64_stage
WORKDIR /app
RUN apt-get update
RUN apt-get -y install python3-confluent-kafka
RUN python3 -m pip install tensorflow===2.8.0 
RUN python3 -m pip install watchdog

COPY /modules /app/modules
COPY /federated /app/federated
COPY main.py /app/main.py

CMD ["python3", "main.py"]

FROM ubuntu:20.04 as arm64_stage
WORKDIR /app
RUN apt-get update
RUN apt-get -y install python3-confluent-kafka
RUN apt-get -y install python3-pip
RUN python3 -m pip install pika===1.2.0
RUN python3 -m pip install tensorflow-aarch64 -f https://tf.kmtea.eu/whl/stable.html
RUN python3 -m pip install watchdog neurokit2

COPY /base /app/base
COPY /modules /app/modules
COPY /federated /app/federated
COPY main.py /app/main.py


CMD ["python3", "main.py"]
