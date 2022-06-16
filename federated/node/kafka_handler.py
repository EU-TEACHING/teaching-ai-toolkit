import socket, os
from pathlib import Path
import logging
import json
from tensorflow import keras
import numpy as np

from confluent_kafka import Consumer, Producer
from confluent_kafka import KafkaError, KafkaException

from base.communication.packet import DataPacket


class KafkaAggregationHandler:

    def __init__(self, params):

        self.receiver = None
        self.producer = None

        self.ds_test = None
        self.receiver_running = True

        self.init_communication(params['broker_addr'], params['groupid'])

    def init_communication(self, broker_addr, groupid):

        # Configure receiver
        conf_receiver = {'bootstrap.servers': broker_addr,
                         'group.id': groupid,
                         'auto.offset.reset': 'smallest'}

        # Configure producer
        conf_producer = {'bootstrap.servers': broker_addr,
                         'client.id': f"{socket.gethostname()}.{os.getenv('SERVICE_NAME')}"}

        # Instantiate producer and receiver
        self.producer = Producer(conf_producer)
        self.receiver = Consumer(conf_receiver)


class KafkaAggregationProducer(KafkaAggregationHandler):

    def __call__(self, model_data: DataPacket) -> None:
        for m_pkt in model_data:
            if 'model' in m_pkt.body:   
                m_pkt.body['model'] = _serialize(m_pkt.body['model'])

            self.producer.produce(m_pkt.topic, value=m_pkt.dumps())
            self.producer.flush()


class KafkaAggregationConsumer(KafkaAggregationHandler):

    def __init__(self, params, topics):
        super(KafkaAggregationConsumer, self).__init__(params)
        self.receiver.subscribe(topics)
        self.params = params

    def __call__(self) -> DataPacket:
        try:
            while self.receiver_running:
                msg = self.receiver.poll(self.params['timeout'])
                if msg is not None and msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        # End of partition event
                        logging.error('%% %s [%d] end at offset %d\n' % (msg.topic(), msg.partition(), msg.offset()))
                    elif msg.error():
                        raise KafkaException(msg.error())
                else:
                    msg = DataPacket.from_json(msg.value()) if msg is not None else None
                    if msg is not None and 'model' in msg.body:
                        msg.body['model'] = _deserialize(msg.body['model'])

                    yield msg

        finally:
            # Close down consumer to commit final offsets.
            self.receiver.close()



def _serialize(model):
    config = model.to_json()
    weights = [w.tolist() for w in model.get_weights()]
    return {'config': config, 'weights': weights}

def _deserialize(model_json):
    model = keras.models.model_from_json(model_json['config'])
    model.set_weights([np.array(w) for w in model_json['weights']])
    return model
