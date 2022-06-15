import pickle
import queue
import threading
from typing import Dict, Iterable

from .node.fednode import FederatedNode
from base.communication.packet import DataPacket


class FederatedClient:

    def __init__(self, learning_module) -> None:
        self._lm = learning_module
        self._topic = f"federated.{self._lm.__class__.FED_TOPIC}"

        self._producer = None
        self._consumer = None
        self._event_queue = queue()
    
    def send_model(self, model_info: Dict):
        self._event_queue.put(model_info)
    
    @FederatedNode(produce=False, consume=True)
    def _consume(self, server_packet_queue: Iterable[DataPacket]) -> Iterable[DataPacket]:

        for m_pkt in server_packet_queue:
            if m_pkt is not None:
                if m_pkt.topic == f"{self._topic}.global_model":
                    m_pkt = self._deserialize(m_pkt)
                    self._lm.model = m_pkt['body']['model']
    
    @FederatedNode(produce=True, consume=False)  
    def _produce(self):
        while True:
            event_dp = self._event_queue.get()
            if 'model' in event_dp['body'] and 'metadata' in event_dp['body']:
                yield DataPacket(
                    topic=f"{self._topic}.local_model",
                    body=self._serialize(event_dp)
                )
    
    def _deserialize(self, model_packet: DataPacket) -> DataPacket:
        model_packet['body']['model'] = pickle.loads(model_packet['body']['model'])
        return model_packet
    
    def _serialize(self, model_packet: DataPacket) -> DataPacket:
        model_packet['body']['model'] = pickle.dumps(model_packet['body']['model'])
        return model_packet
    
    def _build(self):
        self._producer = threading.Thread(target=self._produce, args=(self,))
        self._consumer = threading.Thread(target=self._consume, args=(self,))
        self._producer.start()
        self._consumer.start()