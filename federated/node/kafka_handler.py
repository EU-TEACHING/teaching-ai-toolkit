import socket
import logging

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
                         'client.id': socket.gethostname()}

        # Instantiate producer and receiver
        self.producer = Producer(conf_producer)
        self.receiver = Consumer(conf_receiver)


class KafkaAggregationProducer(KafkaAggregationHandler):

    def __call__(self, model_data: DataPacket) -> None:
        self.producer.produce(model_data.topic, value=model_data.dumps())
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

                if msg is None:
                    yield None

                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        # End of partition event
                        logging.error('%% %s [%d] end at offset %d\n' % (msg.topic(), msg.partition(), msg.offset()))
                    elif msg.error():
                        raise KafkaException(msg.error())
                else:
                    yield DataPacket.from_json(msg.value())

        finally:
            # Close down consumer to commit final offsets.
            self.receiver.close()
