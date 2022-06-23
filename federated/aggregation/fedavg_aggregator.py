import os
import numpy as np

from typing import Optional
from tensorflow import keras
from typing import Dict
from .base_aggregator import FederatedAggregator

from node.communication.serialization import model_from_packet_body, model_to_packet_body
from base.communication.packet import DataPacket

class FedAvgAggregator(FederatedAggregator):

    def __init__(self) -> None:
        self._storage_path = os.getenv('LOCAL_MODELS_STORAGE', 'rl')
        self._client_paths = set()

        self._client_model_ext = os.getenv('CLIENT_MODEL_EXT', 'h5')
        self.n_before_avg = int(os.getenv('N_BEFORE_AVG', '2'))

        os.makedirs(self._storage_path, exist_ok=True)

    def __call__(self, model_packet: DataPacket, **kwargs) -> Optional[Dict]:
        if model_packet is None:
            return None
        
        c_path = os.path.join(self._storage_path, f"{client_id}.{self._client_model_ext}")
        model_packet.body['model'] = model_to_packet_body(model_packet.body['model'])
        model_packet.to_file(c_path)
        self._client_paths.add(c_path)

        if len(self._client_paths) >= self.n_before_avg:
            aggregated = keras.models.model_from_json(model_packet['model']['config'])
            
            new_weights = [np.array(DataPacket.from_file(p).body['model']['weights']) for p in self._client_paths]
            new_weights = sum(new_weights) / len(new_weights) # TODO: edit averaging to take into account dataset size
            aggregated.set_weights(new_weights.tolist())
            self._client_paths = set()
            return {'model': aggregated, 'metadata': {}}
        
        return None
