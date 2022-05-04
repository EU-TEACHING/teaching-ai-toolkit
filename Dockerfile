FROM chronis10/teaching-base:amd64 as amd64_stage
WORKDIR /app
COPY /modules /app/modules
COPY main.py /app/main.py

RUN python3 -m pip install tensorflow===2.8.0;

CMD ["python3", "main.py"]

FROM armswdev/tensorflow-arm-neoverse:r22.04-tf-2.8.0-eigen as arm64_stage
WORKDIR /app
COPY /base/communication /app/communication
COPY /base/node.py /app/node.py
COPY /modules /app/modules
COPY main.py /app/main.py

RUN python3 -m pip install pika===1.2.0

CMD ["python3", "main.py"]