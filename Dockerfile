FROM armswdev/tensorflow-arm-neoverse:r22.04-tf-2.8.0-eigen
WORKDIR /app
COPY /base/communication /app/communication
COPY /base/node.py /app/node.py
COPY /modules /app/modules
COPY main.py /app/main.py

RUN pip3 install pika===1.2.0

CMD ["python3", "main.py"]