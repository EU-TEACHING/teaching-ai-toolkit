import os
import pickle
from typing import Iterable

from base.communication.packet import DataPacket
from .node.fednode import FederatedNode

class FederatedServer:

    def __init__(self) -> None:
        self._aggregation_method = os.getenv('AGGREGATION', 'fedavg')
        self._topic = os.getenv('TOPIC')
        self._aggregator = None
    

    @FederatedNode(produce=True, consume=True)
    def __call__(self, model_packet_queue: Iterable[DataPacket]) -> Iterable[DataPacket]:
        for m_pkt_ser in model_packet_queue:
            if m_pkt_ser is None:
                aggregated = self._aggregator(None)
            else:
                m_pkt = self._deserialize(m_pkt_ser)
                aggregated = self._aggregator(m_pkt['body']['model'], metadata=m_pkt['body']['metadata'])

            if aggregated is not None:
                yield self._serialize(DataPacket(
                    topic=self._topic,
                    body=aggregated
                ))
            
    def _deserialize(self, model_packet: DataPacket) -> DataPacket:
        model_packet['body']['model'] = pickle.loads(model_packet['body']['model'])
        return model_packet
    
    def _serialize(self, model_packet: DataPacket) -> DataPacket:
        model_packet['body']['model'] = pickle.dumps(model_packet['body']['model'])
        return model_packet
    
    def _build(self) -> None:
        pass
        