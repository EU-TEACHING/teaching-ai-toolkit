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
            if self.buffer.ready:
                x, y = self.buffer.samples
                self._model.fit(x, y, **kwargs)
            else:
                print("Buffer not ready.")
        else:
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
        self._feature_keys = os.environ['FEATURE_KEYS'].split(',')
        self._target_key = os.getenv('TARGET_KEY')

        self._x = []
        if self._target_key is not None:
            self._y = []

    def __call__(self, msg) -> None:
        self._x += [[msg.body[k] for k in self._feature_keys]] if not isinstance(msg.body, List) else [[b[k] for k in self._feature_keys] for b in msg.body]
        if self._target_key is not None:
            self._y += [msg.body[self._target_key]] if not isinstance(msg.body, List) else [b[self._target_key] for b in msg.body]
        if len(self) >= self._max_size:
            if self._incremental:
                self._x = self._x[(len(self) - self._max_size):]
                self._y = self._y[(len(self) - self._max_size):]
            else:
                self._x = self._x[:(len(self) - self._max_size)]
                self._y = self._y[:(len(self) - self._max_size)]

    def __len__(self):
        return len(self._x)
    
    def flush(self, start: int = 0, end: int = None):
        if end is None:
            end = len(self)
        self._x = self._x[start:end]
        self._y = self._y[start:end]
    
    @property
    def samples(self):
        if self._target_key is None:
            return np.array(self._x), None
        else:
            return np.array(self._x), np.array(self._y)
    
    @property
    def ready(self):
        return len(self) > self._min_size
