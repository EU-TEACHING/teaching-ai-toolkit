import os

import tensorflow as tf
import numpy as np

from base.node import TEACHINGNode
from base.communication.packet import DataPacket
from .base_module import LearningModule



class RLModule(LearningModule):

    def __init__(self):
        super(RLModule, self).__init__()
        self._model_path = os.getenv('MODEL_PATH')
        self._build()
        self._aggregator = Aggregator()
        
    @TEACHINGNode(produce=True, consume=True)
    def __call__(self, input_fn):
        
        for msg in input_fn:
            
            self._aggregator.aggregate(msg)
            if self._aggregator.is_ready():
                profile = self._model.predict(np.asarray([self._aggregator._batch_data]))
                self._aggregator.clean()
                final_value = float(np.argmax(profile[0]))
                yield DataPacket(
                    topic='prediction.driving_profile.value', 
                    #timestamp= msg.timestamp,                    
                    body={'driving_profile': final_value })

    def _build(self):
        self._model = tf.keras.models.load_model(self._model_path)
        self._model.summary()


class Aggregator():

    def __init__(self):
        self._namespaces = ["stress", "excitement","ay","gz","speed","speed_limit"]
        self._batch_data = [None]*len(self._namespaces)
        
    def aggregate(self,msg):
        if type(msg) == list:
            msg = msg[0]
        msg_keys = msg.body.keys()        
        for vkey in msg_keys:
            if vkey in self._namespaces:
                position =  self._namespaces.index(vkey)
                self._batch_data[position] = msg.body[vkey]
        

    def is_ready(self):        
        for value in self._batch_data:
            if value is None:
                return False
        return True

    def clean(self):
        self._batch_data = [None]*len(self._namespaces)



