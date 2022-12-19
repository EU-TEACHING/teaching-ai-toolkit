import os
import threading

import tensorflow as tf
from tensorflow import keras
import numpy as np
import time
from collections import deque

from base.node import TEACHINGNode
from base.communication.packet import DataPacket
from .base_module import LearningModule

class RLModule(LearningModule):

    FED_TOPIC='rlmodule'

    def __init__(self):
        super(RLModule, self).__init__()
        self._periodic_sender = None
        self._aggregator = None
        self._trainnable = False
        self._personalizer = None
        self._build()

    @TEACHINGNode(produce=True, consume=True)
    def __call__(self, input_fn):
        for msg in input_fn:
            self._aggregator.aggregate(msg)
            if self._aggregator.is_ready():
                #profile = self._model.predict(np.asarray([self._aggregator._batch_data]))
                obs = np.asarray([self._aggregator._batch_data])
                actions, value = self.model.action_value(np.expand_dims(obs, axis=0))
                if self._trainnable:
                     self._personalizer.store_data(obs,np.argmax(actions[0]),value)
                     if self._personalizer._new_obs_counter>= self._personalizer._batch_sz:
                        self._personalizer.start_trainning()
                self._aggregator.clean()
                final_value = float(np.argmax(actions[0]))
                yield DataPacket(
                    topic='prediction.driving_profile.value',
                    body={'driving_profile': final_value }
                )

    def _build(self):
        super(RLModule, self)._build()
        self._model = tf.keras.models.load_model(self._model_path)
        self._model.summary()
        self._aggregator = Aggregator()
        if self.federated:
            self._periodic_sender = periodic_send_model(self)
        if self._trainnable:
            self._personalizer = Personalize(self._model)

    def _train(self):
        pass

class Personalize():
    def __init__(self,model):
        self._gamma = 0.99
        self._value_c = 0.55
        self._entropy_c = 1e-4
        self._batch_sz = 250
        self._observation_space = (1,6)
        self._actions = 3
        self._actions = np.empty((self._batch_sz,), dtype=np.int32)
        self._rewards, self._dones, self._values = np.empty((self._actions, self._batch_sz,))
        self._observations = np.empty((self._batch_sz,)+self._observation_space)
        self.model = model
        self._obs_storage = deque(maxlen=self._batch_sz)
        self._actions_storage = deque(maxlen=self._batch_sz)
        self._values_storage = deque(maxlen=self._batch_sz)
        self._new_obs_counter = 0

    def store_data(self,obs,action,values):
        self._obs_storage.append(obs)
        self._actions_storage.append(action)
        self._values_storage.append(values)
        self._new_obs_counter +=1

    def _generate_batch(self):
        for i in range(1,self._batch_sz):
            self._observations[i] = self._obs_storage[i-1]
            self._actions[i] = self._actions_storage[i]
            self._values[i] = self._values_storage[i]
            reward,done = self._calculate_reward(self._obs_storage[i])
            self._rewards[i] = reward
            self._dones[i] = done

    def start_trainning(self):
        self._new_obs_counter = 0
        tranning_thread = threading.Thread(target=self._train_on_batch, daemon=True)
        tranning_thread.start()

    def _calculate_reward(self,obs):
        #  Reward based on Stress
        reward = 1
        #  Done status based on a ultimate goal
        done = 0
        return reward,done

    def _train_on_batch(self):
        self._generate_batch()
        _, next_value = self.model.action_value(self._obs_storage[None,:])
        returns, advs = self._returns_advantages(self._rewards, self._dones, self._values, next_value)
        acts_and_advs = np.concatenate([self._actions[:, None], advs[:, None]], axis=-1)
        losses = self.model.network.train_on_batch(self._observations, [acts_and_advs, returns])
        self.model.network.save('models/new_model.h5')

    def _returns_advantages(self, rewards, dones, values, next_value):
        returns = np.append(np.zeros_like(rewards), next_value, axis=-1)
        for t in reversed(range(rewards.shape[0])):
            returns[t] = rewards[t] + self._gamma * returns[t + 1] * (1 - dones[t])
        returns = returns[:-1]
        advantages = returns - values
        return returns, advantages

    def _value_loss(self, returns, value):
        return self._value_c * keras.losses.mean_squared_error(returns, value)

    def _logits_loss(self, actions_and_advantages, logits):
        actions, advantages = tf.split(actions_and_advantages, 2, axis=-1)
        weighted_sparse_ce = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        actions = tf.cast(actions, tf.int32)
        policy_loss = weighted_sparse_ce(actions, logits, sample_weight=advantages)
        probs = tf.nn.softmax(logits)
        entropy_loss = keras.losses.categorical_crossentropy(probs, probs)
        return policy_loss - self._entropy_c * entropy_loss

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

def periodic_send_model(lm):
    def aux_fn():
        send_every_t = int(os.getenv('SEND_MODEL_INTERVAL', '5'))
        while True:
            time.sleep(send_every_t)
            lm._train()
            lm._client.send_model = {'model': tf.keras.models.clone_model(lm._model), 'metadata': {}}
   
    periodic_sender = threading.Thread(target=aux_fn)
    periodic_sender.start()
    return periodic_sender