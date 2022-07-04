import os
import threading
from typing import Iterable

from .node.fednode import FederatedNode

from base.communication.packet import DataPacket


class FederatedClient(threading.Thread):

    def __init__(self, learning_module) -> None:
        threading.Thread.__init__(self)
        self._lm = learning_module
        self._topic = f"federated.{self._lm.__class__.FED_TOPIC}"

        self._producer = None
        self._consumer = None
        self.send_model = None
    
    @FederatedNode(produce=True, consume=True)
    def run(self, server_packet_queue: Iterable[DataPacket]) -> Iterable[DataPacket]:
        for m_pkt in server_packet_queue:
            if m_pkt is not None:
                if f"{self._topic}.global_model" == m_pkt.topic:
                    print(f"Client {os.environ['SERVICE_NAME']} received a new global model.", flush=True)
                    self._lm.model = m_pkt.body['model']

            if self.send_model is not None:
                print(f"Client {os.environ['SERVICE_NAME']} is sending a new local model.", flush=True)
                yield DataPacket(
                    topic=f"{self._topic}.local_model",
                    body=self.send_model
                )
                self.send_model = None
    
    def get_subscribe_topics(self):
        return [f"{self._topic}.global_model"]
  