import os
from typing import Iterable

from base.communication.packet import DataPacket
from .node.fednode import FederatedNode
from .aggregation.aggregators import get_aggregator


class FederatedServer:

    def __init__(self) -> None:
        self._aggregation_method = os.getenv('AGGREGATION', 'fedavg')
        self._topic = f"federated.{os.getenv('MODEL_TOPIC')}"
        self._aggregator = None
        self._build()
        print("Federated server instantiated.", flush=True)
    

    @FederatedNode(produce=True, consume=True)
    def __call__(self, model_packet_queue: Iterable[DataPacket]) -> Iterable[DataPacket]:
        for m_pkt in model_packet_queue:
            if m_pkt is not None:
                print("Server received a local model from", m_pkt.service_name, flush=True)
            aggregated = self._aggregator(m_pkt)

            if aggregated is not None:
                print("Server is broadcasting a new global model.", flush=True)
                yield DataPacket(
                    topic=f"{self._topic}.global_model",
                    body=aggregated
                )

    def _build(self) -> None:
        self._aggregator = get_aggregator(self._aggregation_method)
    
    def get_subscribe_topics(self):
        return [f"{self._topic}.local_model"]
        