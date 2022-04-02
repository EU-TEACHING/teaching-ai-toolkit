import os
import json

import tensorflow_addons as tfa
from tensorflow import keras

from base.node import TEACHINGNode
from base.communication.packet import DataPacket
from .base_module import LearningModule


class StressModule(LearningModule):

    def __init__(self):
        super(self, StressModule).__init__()
        self._model_path = os.getenv('MODEL_PATH')
        self._hparams_path = os.environ['HPARAMS_PATH'] if self._model_path is None else None
    
    @TEACHINGNode(produce=True, consume=True)
    def __call__(self, input_fn):

        for msg in input_fn:
            x = msg # TODO: process the message as input of the model
            stress_value = self._model(x)
            yield DataPacket(topic='stress', body={'value': stress_value})

    def _build(self):
        if self._file_path is not None:
            self._model = keras.models.load_model(self._model_path)
        else:
            config = json.loads(open(self._hparams_path).read())
            inputs = keras.Input(shape=(config['INPUT_SIZE'], None))
            for i in range(config['LAYERS']):
                x = tfa.layers.ESN(
                    units=config['UNITS'],
                    connectivity=config['CONNECTIVITY'],
                    leaky=config['LEAKY'],
                    spectral_radius=config['RHO'],
                    return_sequences=True,
                )(inputs if i == 0 else x)
            outputs = keras.layers.Dense(
                config['CLASSES'], 
                activation=('sigmoid' if config['CLASSES'] <= 2 else 'softmax')
            )(x)
            self._model = keras.Model(inputs=inputs, outputs=outputs, name="stress_model")

        self._model.summary()
