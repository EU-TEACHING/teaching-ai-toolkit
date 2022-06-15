import os
import logging
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from typing import Optional
from tensorflow import keras
from pathlib import Path
from typing import Dict
from .base_aggregator import FederatedAggregator

class FedAvgAggregator(FederatedAggregator):

    def __init__(self) -> None:
        self._client_queues = []
        self._local_store = []
        self.model_common = None

        self.NUM_MSGS = int(os.environ['NUM_MSG'])
        self.client_model_prefix = os.environ['client_model_prefix']
        self.client_model_ext = os.environ['client_model_ext']

        # self.incoming_path = incomingpath
        self.outgoing_path = os.environ['outgoing_path']

        # hack for the second integration demo, it will contain the last seen Sender ID, used in asnwering back
        self.SenderId = "X"


    # create a model from the weights of multiple models
    def model_weight_ensemble(self, members):
        # determine how many layers need to be averaged
        n_layers = len(members[0].get_weights())

        # create an set of average model weights
        avg_model_weights = list()

        for layer in range(n_layers):
            # collect this layer from each model
            layer_weights = np.array([model.get_weights()[layer] for model in members])

            # weighted average of weights for this layer
            avg_layer_weights = np.average(layer_weights, axis=0)

            # store average layer weights
            avg_model_weights.append(avg_layer_weights)

        # create a new model with the same structure
        model = keras.models.clone_model(members[0])

        # set the weights in the new
        model.set_weights(avg_model_weights)

        return model


    # load in memory all models from files referenced in the local storage (local dir), return a list of models
    def load_all_client_models(self, n_start, n_end):
        all_client_models = list()

        for epoch in range(n_start, n_end):
            # define filename for this ensemble
            filename = self.client_model_prefix + "_" + str(epoch) + "." + self.client_model_ext

            # we may just read the whole model, but this may change again in later revisions

            # load model from file
            model = keras.models.clone_model(self.model_common)

            # load weights
            model.load_weights(filename)

            # add to list of members
            all_client_models.append(model)

        return all_client_models


    # process function for a single model received;
    # return averaged models data if it is generated, otherwise return None
    def process_model(self, model):
        # we get a filename naw, we nee to parse it to extract the sender id as well as copy it to our private storage
        # we will need to rework the management to avoid copying twice the files
        # we will need in the future to add more metadata, like timestamps


        # choose a filename in the local store, copy the received file message there
        filename = f'{self.client_model_prefix}_{len(self._local_store)}.{self.client_model_ext}'

        # saving model binary content to the disk
        model.save(filename)
        # self.write_modelfile(filename, model)
        self._local_store.append(filename)

        if self.model_common is None:
            self.model_common=model

        # do nothing: the model remains in the storage volume and will be evaluated later on

        logging.info(f'Model received (possibly encrypted), length {len(model)}')

        if len(self._local_store) >= self.NUM_MSGS:

            logging.info(f'Average to be computed on {len(self._local_store)} models')

            # reference https://machinelearningmastery.com/polyak-neural-network-model-weight-ensemble/
            members = self.load_all_client_models(0, self.NUM_MSGS)
            averaged = self.model_weight_ensemble(members)

            #  we are not using the local store indeed, clear it for the side effect of resetting the file names
            self._local_store.clear()
            #  return the averaged model to the caller when we produce one
            return averaged

    def __call__(self, model: keras.Model, **metadata) -> Optional[Dict]:
        if model is None:
            return None
        
        # Apply FedAvg
        return self.process_model(model)