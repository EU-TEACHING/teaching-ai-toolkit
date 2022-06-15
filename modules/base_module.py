import os
import time
import threading

import numpy as np

from federated.client import FederatedClient


class LearningModule(object):

    def __init__(self):
        self._model_path = os.getenv('MODEL_PATH')

        self._exec_mode = os.getenv('MODE')
        if self._exec_mode == 'FEDERATED':
            self._client = None
            self._sender = None
        
        self._model = None
        self._phase = 'eval'
        

    def __call__(self, input_fn):
        raise NotImplementedError
            
    def _train(self):
        weights = self._model.get_weights()
        weights = [np.random.permutation(w.flat).reshape(w.shape) for w in weights]
        self._model.set_weights(weights)
    
    def _build(self):
        if self._exec_mode == 'FEDERATED':
            self._client = FederatedClient(self)
            self._client._build()
            self._sender = periodic_sender(self)
            self._sender.start()
    
    @property
    def phase(self):
        return self._phase
    
    @phase.setter
    def phase(self, new_phase):
        if type(new_phase) != str:
            raise TypeError('Phase new_phase must be a string.')
        if not new_phase in ['train', 'eval']:
            raise ValueError('Phase new_phase must be in {"train", "eval"}') 
        
        self._phase = new_phase
    
    @property
    def model(self):
        return self._model
    
    @model.setter
    def model(self, new_model):
        self._model = new_model


def periodic_sender(lm):
    def _periodic_model_update(lm):
        send_every_t = int(os.getenv('SEND_MODEL_INTERVAL', '10'))
        last_new_model_t = time.time()
        while True:
            if (time.time() - last_new_model_t) > send_every_t:
                lm._train()
                last_new_model_t = time.time()
                lm._client.send_model({'model': lm.model, 'metadata': {}})
    
    return threading.Thread(target=_periodic_model_update, args=(lm, ))