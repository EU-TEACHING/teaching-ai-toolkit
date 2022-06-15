from typing import Optional
from tensorflow import keras

from .base_aggregator import FederatedAggregator


class FedAvgAggregator(FederatedAggregator):

    def __init__(self) -> None:
        self._client_queues = []

    def __call__(self, model: keras.Model, **metadata) -> Optional[keras.Model]:
        if model is None:
            return None
        
        # Apply FedAvg
        pass