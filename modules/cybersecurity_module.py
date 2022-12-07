import os
import time
import threading
from typing import List

import pandas as pd
import tensorflow as tf
from tensorflow import keras
import numpy as np

from base.node import TEACHINGNode
from base.communication.packet import DataPacket
from .base_module import LearningModule

from cybersecurity.src.inference.inferrer import Inferrer
from cybersecurity.src.configs.config import CFG


class CybersecurityModule(LearningModule):
    FED_TOPIC = 'cybersecuritymodule'

    def __init__(self):
        super(CybersecurityModule, self).__init__()
        self._build()
        self._seq_time_steps = int(os.getenv('SEQ_TIME_STEPS'))
        self.infer = Inferrer(CFG)

    @TEACHINGNode(produce=True, consume=True)
    def __call__(self, input_fn):
        queue = []
        for msg in input_fn:
            queue.append(msg.body)
            if len(queue) == self._seq_time_steps:
                seq_df = pd.DataFrame(queue)
                self.infer.load_data_online(seq_df)
                pred_df = self.infer.predict()

                yield DataPacket(
                    topic='prediction.cybersecurity.value',
                    timestamp=msg.timestamp,
                    body=pred_df)

            queue.pop()

    def _build(self):
        super(CybersecurityModule, self)._build()
        if self._model_path is not None and os.path.exists(self._model_path):
            model_path = os.path.join(self._model_path, os.getenv('MODEL_ID'))
            transformer_path = os.path.join(self._model_path, os.getenv('TRANSFORMER_ID'))

            self._model, self._transformer = self.infer.load_model_online(model_path, transformer_path)
            self._model.summary()
        else:
            print("Trained model was not found in self._model_path")
