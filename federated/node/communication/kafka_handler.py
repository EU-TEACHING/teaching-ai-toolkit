import socket, os
import logging
from typing import Dict, Iterator, List
import keras
from confluent_kafka import Consumer, Producer
from confluent_kafka import KafkaError, KafkaException

from .serialization import model_from_packet_body, model_to_packet_body
from base.communication.packet import DataPacket


class KafkaAggregationProducer:

    def __init__(self, params: Dict) -> None:
        self._config = {
            'bootstrap.servers': params['broker_addr'],
            'client.id': f"{socket.gethostname()}.{os.getenv('SERVICE_NAME')}"
        }
        self.producer = Producer(self._config)
    
    def __call__(self, model_data: DataPacket) -> None:
        for m_pkt in model_data:
            if 'model' in m_pkt.body and isinstance(m_pkt.body['model'], keras.Model):   
                m_pkt.body['model'] = model_to_packet_body(m_pkt.body['model'])

            self.producer.produce(m_pkt.topic, value=m_pkt.dumps())
            self.producer.flush()


class KafkaAggregationConsumer:

    def __init__(self, params: Dict, topics: List[str]):
        self._config = {
            'bootstrap.servers': params['broker_addr'],
            'group.id': params['groupid'],
            'auto.offset.reset': 'smallest'
        }
        self._timeout = params['timeout']
        
        self.consumer = Consumer(self._config)
        self.consumer.subscribe(topics)

    def __call__(self) -> Iterator[DataPacket]:
        try:
            while True:
                msg = self.consumer.poll(self._timeout)
                if msg is not None and msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        # End of partition event
                        logging.error('%% %s [%d] end at offset %d\n' % (msg.topic(), msg.partition(), msg.offset()))
                    elif msg.error():
                        raise KafkaException(msg.error())
                else:
                    msg = DataPacket.from_json(msg.value()) if msg is not None else None
                    if msg is not None and 'model' in msg.body:
                        msg.body['model'] = model_from_packet_body(msg.body['model'])

                    yield msg

        finally:
            self.consumer.close()
