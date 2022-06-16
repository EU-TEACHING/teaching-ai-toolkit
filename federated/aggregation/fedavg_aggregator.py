import os
import numpy as np

from typing import Optional
from tensorflow import keras
from typing import Dict
from .base_aggregator import FederatedAggregator

class FedAvgAggregator(FederatedAggregator):

    def __init__(self) -> None:
        self._storage_path = os.getenv('LOCAL_MODELS_STORAGE', 'rl')
        self._client_paths = set()

        self._client_model_ext = os.getenv('CLIENT_MODEL_EXT', 'h5')
        self.n_before_avg = int(os.getenv('N_BEFORE_AVG', '2'))

        os.makedirs(self._storage_path, exist_ok=True)

    def __call__(self, model: keras.Model, client_id: Optional[str] = None, **metadata) -> Optional[Dict]:
        if model is None:
            return None
        
        c_path = os.path.join(self._storage_path, f"{client_id}.{self._client_model_ext}")
        model.save(c_path)
        self._client_paths.add(c_path)

        if len(self._client_paths) >= self.n_before_avg:
            aggregated = keras.models.model_from_json(model.to_json())
            new_weights = [np.array(keras.models.load_model(p).get_weights()) for p in self._client_paths]
            new_weights = sum(new_weights) / len(new_weights) # TODO: edit averaging to take into account dataset size
            aggregated.set_weights(new_weights.tolist())
            self._client_paths = set()
            return {'model': aggregated, 'metadata': {}}
        
        return None
