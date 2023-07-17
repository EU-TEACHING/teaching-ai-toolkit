from tensorflow import keras
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import numpy as np
import time

from base.node import TEACHINGNode
from .base_module import LearningModule


class ORModule(LearningModule):

    def __init__(self):

        self.model = None
        self._build()
        with open("/app/storage/time.log", "a") as f:
                           f.write("----Tensorflow-----\n")
    
    @TEACHINGNode(produce=False, consume=True)
    def __call__(self, input_fn):
        for msg in input_fn:
            start_time = time.time()
            np_img = np.asarray(eval(msg.body["img"]), dtype='uint8').reshape(224, 224, 3)
            print("image received!")

            img_batch = np.expand_dims(np_img, 0)
            pred = self.model.predict(preprocess_input(img_batch))
            end_time = time.time()
            execution_time = end_time - start_time
            with open("/app/storage/time.log", "a") as f:
                           f.write(f"{execution_time}\n")
            print("Predictions: ", decode_predictions(pred))

    def _build(self):
        self.model = keras.applications.MobileNetV2(weights='imagenet', include_top=True)
        self.model.trainable = False
        self.model.summary()
