import os
from typing import Iterable

from base.communication.packet import DataPacket
from .node.fednode import FederatedNode
from .aggregation.aggregators import get_aggregator

class FederatedServer:

    def __init__(self) -> None:
        self._aggregation_method = os.getenv('AGGREGATION', 'fedavg')
        self._model_topic = os.getenv('MODEL_TOPIC')
        self._aggregator = None
        self._build()
        print("Federated server instantiated.", flush=True)
    

    @FederatedNode(produce=True, consume=True)
    def __call__(self, model_packet_queue: Iterable[DataPacket]) -> Iterable[DataPacket]:
        for m_pkt in model_packet_queue:
            if m_pkt is None:
                aggregated = self._aggregator(None)
            else:
                print("Server received a local model from", m_pkt.service_name, flush=True)
                aggregated = self._aggregator(m_pkt.body['model'], m_pkt.service_name, metadata=m_pkt.body['metadata'])

            if aggregated is not None:
                print("Server is broadcasting a new global model.", flush=True)
                yield DataPacket(
                    topic=self._model_topic,
                    body=aggregated
                )

    def _build(self) -> None:
        self._aggregator = get_aggregator(self._aggregation_method)
        