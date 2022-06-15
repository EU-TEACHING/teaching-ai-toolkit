import os
import time
from datetime import datetime

import numpy as np

from base.communication.packet import DataPacket

class LearningModule(object):

    def __init__(self):
        self._warm_start_path = os.getenv('WARM_START_PATH')
        self._predictions_topic = os.getenv('PREDICTIONS_TOPIC')

        self._exec_mode = os.getenv('MODE')
        if self._exec_mode == 'FEDERATED':
            self._model_info_topic = os.getenv('MODEL_INFO_TOPIC')

            self._local_models_path = os.getenv('LOCAL_MODELS_PATH')       
            self._federated_models_path = os.getenv('FEDERATED_MODELS_PATH')
            self._federated_model_topic = os.getenv('FEDERATED_MODEL_TOPIC')
            self._save_every_t = int(os.getenv('SAVE_MODEL_INTERVAL', '10'))
        
        self._model = None
        self._phase = 'eval'
        self._last_new_model_t = time.time()
    
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
    
    def _handle_generic_lm_packet(self, msg):
        if self._exec_mode == 'FEDERATED' and msg.topic == self._federated_model_topic:
            self._model.load_model(msg.body['path'])

    def _periodic_model_update(self):
        if self._exec_mode == 'FEDERATED' and (time.time() - self._last_new_model_t) > self._save_every_t:
            path = self._train()
            self._last_new_model_t = time.time()
            return DataPacket(
                topic=self._model_info_topic,                 
                body={'path': path }    
            )
        return None
            
    def _train(self):
        weights = self._model.get_weights()
        weights = [np.random.permutation(w.flat).reshape(w.shape) for w in weights]
        self._model.set_weights(weights)
        file_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = f'{self._local_models_path}/{file_timestamp}_model.h5'
        self._model.save(path)
        return path
    
    def _build(self):
        if self._exec_mode == 'FEDERATED':
            os.makedirs(self._federated_models_path, exist_ok=True)

    def __call__(self, input_fn):
        raise NotImplementedError