from tensorflow import keras
from keras.applications.efficientnet import decode_predictions
import numpy as np

from base.node import TEACHINGNode
from .base_module import LearningModule


class ORModule(LearningModule):

    def __init__(self):

        self.model = None
        self._build()
    
    @TEACHINGNode(produce=False, consume=True)
    def __call__(self, input_fn):
        for msg in input_fn:
            np_img = msg.body["img"]
            print("image received!")

            img_batch = np.expand_dims(np_img, 0)
            pred = self.model.predict(img_batch)
            print("Predictions: ", decode_predictions(pred))

    def _build(self):
        self.model = keras.applications.EfficientNetV2B0(weights='imagenet', include_top=True)
        self.model.trainable = False
        self.model.summary()