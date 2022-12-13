import os
from typing import Any, List
import numpy as np

from federated.client import FederatedClient


class LearningModule(object):

    def __init__(self):
        self._model_path = os.getenv('MODEL_PATH')

        self._exec_mode = os.getenv('MODE')
        if self._exec_mode == 'FEDERATED':
            self._client = None
        
        self._model = None
        self._trainable = os.getenv('TRAINABLE', 'false').lower() == 'true'
        if self._trainable:
            self.buffer = TrainingBuffer()

    def __call__(self, input_fn):
        raise NotImplementedError
            
    def fit(self, x=None, y=None, **kwargs):
        if x is None:
            x = self.buffer.samples
        self._model.fit(x, y, **kwargs)
    
    def _build(self):
        if self.federated:
            self._client = FederatedClient(self)
            self._client.start()

    @property
    def federated(self):
        return self._exec_mode == 'FEDERATED'
    
    @property
    def model(self):
        return self._model
    
    @model.setter
    def model(self, new_model):
        self._model = new_model


class TrainingBuffer(object):

    def __init__(self) -> None:
        self._min_size = int(os.getenv('BUFFER_MIN_SIZE', '1000'))
        self._max_size = int(os.getenv('BUFFER_MAX_SIZE', '10000'))
        self._incremental = os.getenv('INCREMENTAL', 'false').lower() == 'true'
        self._keys_order = os.environ['BUFFER_KEYS_ORDER'].split(',')

        self._window = []

    def __call__(self, msg) -> None:
        self._window += [[msg.body[k] for k in self._keys_order]] if not isinstance(msg.body, List) else [[b[k] for k in self._keys_order] for b in msg.body]
        if len(self) >= self._max_size:
            if self._incremental:
                self._window = self._window[(len(self) - self._max_size):]
            else:
                self._window = self._window[:(len(self) - self._max_size)]

    def __len__(self):
        return len(self._window)
    
    def flush(self, start: int = 0, end: int = None):
        if end is None:
            end = len(self)
        self._window = self._window[start:end]
    
    @property
    def samples(self):
        return np.stack(self._window, axis=0)
    
    @property
    def ready(self):
        return len(self) > self._min_size
