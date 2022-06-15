FROM chronis10/teaching-base:amd64 as amd64_stage
WORKDIR /app
RUN apt-get -y install python3-confluent-kafka
RUN python3 -m pip install tensorflow===2.8.0 

COPY /modules /app/modules
COPY /federated /app/federated
COPY main.py /app/main.py

CMD ["python3", "main.py"]

FROM armswdev/tensorflow-arm-neoverse:r22.04-tf-2.8.0-eigen as arm64_stage
WORKDIR /app
RUN apt-get -y install python3-confluent-kafka
RUN python3 -m pip install pika===1.2.0

COPY /base /app/base
COPY /modules /app/modules
COPY /federated /app/federated
COPY main.py /app/main.py


CMD ["python3", "main.py"]
