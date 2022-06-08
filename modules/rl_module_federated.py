import os
import time
from datetime import datetime
import tensorflow as tf
import numpy as np
from base.node import TEACHINGNode
from base.communication.packet import DataPacket
from .base_module import LearningModule

class RLModule(LearningModule):

    def __init__(self):
        super(RLModule, self).__init__()
        self._initial_model_path = os.getenv('INITIAL_MODEL_PATH')
        self._local_models_path = os.getenv('LOCAL_MODELS_PATH')
        self._predictions_topic = os.getenv('PREDICTIONS_TOPIC')        
        self._federated_models_path = os.getenv('FEDERATED_MODELs_PATH')
        self._new_model_interval = int(os.getenv('NEW_MODEL_INTERVAL'))
        self._federated_model_topic = os.getenv('FEDERATED_MODEL_TOPIC')
        self._local_model_topic = os.getenv('LOCAL_MODEL_TOPIC')
        self._build()
        self._aggregator = Aggregator()
        self._last_new_model = time.time()

        
    @TEACHINGNode(produce=True, consume=True)
    def __call__(self, input_fn):
        
        for msg in input_fn:            
            if  msg.topic.split('.')[0] == 'sensor':
                self._aggregator.aggregate(msg)
                if self._aggregator.is_ready():
                    profile = self._model.predict(np.asarray([self._aggregator._batch_data]))
                    self._aggregator.clean()
                    final_value = float(np.argmax(profile[0]))
                    yield DataPacket(
                        topic=self._predictions_topic , 
                        #timestamp= msg.timestamp,                    
                        body={'driving_profile': final_value })
            elif msg.topic == self._federated_model_topic:
                self.model.load_model(msg.body['path'])

            if (time.time() - self._last_new_model) > self._new_model_interval:
                path = self._train()
                self._last_new_model = time.time()
                packet = DataPacket(
                        topic=self._local_model_topic,                 
                        body={'path': path })                 
                yield packet



    #dummy for now
    def _train(self):
        weights = self._model.get_weights()
        weights = [np.random.permutation(w.flat).reshape(w.shape) for w in weights]
        self._model.set_weights(weights)
        file_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = f'{self._local_models_path}/{file_timestamp}_model.h5'
        self._model.save(path)
        return path 


    def _build(self):
        self._model = tf.keras.models.load_model(self._initial_model_path)
        self._model.summary()

    


class Aggregator():    
    def __init__(self):
        self._namespaces = ["stress", "excitement","ay","gz","speed","speed_limit"]
        self._batch_data = [None]*len(self._namespaces)
        
    def aggregate(self,msg):
        if type(msg.body) == list:
            msg.body = msg.body[0]
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



