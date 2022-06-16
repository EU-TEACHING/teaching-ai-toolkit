from typing import Dict, Optional
from tensorflow import keras


class FederatedAggregator:

    def __init__(self) -> None:
        pass

    def __call__(self, model: keras.Model, client_id: str, **metadata) -> Optional[Dict]:
        """A function that, given a model, applies the aggregation technique

        Args:
            model_packet (DataPacket): a DataPacket where the body contains keys 'model' and 'metadata' which
                contains all the useful information for the aggregator from the client. If None, it applies default
                techniques (e.g., aggregate regardless of new clients after timeout).
            client_id (str): string representing the client who sent the current model
            metadata: additional parameters required for applying the aggregation method

        Raises:
            NotImplementedError: to be implemented by subclasses

        Returns:
            Optional[keras.Model]: return a dictionary with keys 'model' and 'metadata whenever 
                a new model must be broadcasted to all the desired clients
        """
        raise NotImplementedError