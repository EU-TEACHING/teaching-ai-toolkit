import os
import numpy as np

from typing import Optional
from tensorflow import keras
from typing import Dict
from .base_aggregator import FederatedAggregator

from ..node.communication.serialization import model_to_packet_body
from base.communication.packet import DataPacket

class FedAvgAggregator(FederatedAggregator):

    def __init__(self) -> None:
        self._storage_path = os.getenv('LOCAL_MODELS_STORAGE', 'storage/federated/local/rl')
        self._client_paths = set()

        self._client_model_ext = os.getenv('CLIENT_MODEL_EXT', 'dat')
        self.n_before_avg = int(os.getenv('N_BEFORE_AVG', '2'))

        os.makedirs(self._storage_path, exist_ok=True)

    def __call__(self, model_packet: DataPacket, **kwargs) -> Optional[Dict]:
        if model_packet is None:
            return None
        
        c_path = os.path.join(self._storage_path, f"{model_packet.service_name}")
        model = model_packet.body.pop('model')
        model_packet.to_file(f"{c_path}.dat")
        model.save(f"{c_path}.h5")
        self._client_paths.add(c_path)

        if len(self._client_paths) >= self.n_before_avg:
            aggregated = keras.models.model_from_json(model.to_json())
            new_weights = [np.array(keras.models.load_model(f"{p}.h5").get_weights()) for p in self._client_paths]
            new_weights = sum(new_weights) / len(new_weights) # TODO: edit averaging to take into account dataset size
            aggregated.set_weights(new_weights.tolist())
            self._client_paths = set()
            return {'model': aggregated, 'metadata': {}}
        
        return None
