from ast import literal_eval
import tflite_runtime.interpreter as tflite
import numpy as np
import time
import os

from base.node import TEACHINGNode
from .base_module import LearningModule


class ORModule(LearningModule):

    def __init__(self):

        self.model = None
        self._build()
        with open("/app/storage/time.log", "a") as f:
                           f.write("------\n")
                           f.write(f"Model:{self._model_path}\n")
                           f.write(f"Delegate:{self._ext_delegate}\n")
                           f.write(f"Use Float16:{self._use_float16}\n")
    
    @TEACHINGNode(produce=False, consume=True)
    def __call__(self, input_fn):
        for msg in input_fn:
            
            np_img = np.asarray(eval(msg.body["img"]), dtype='uint8').reshape(224, 224, 3)
            print("image received!")

            img_batch = np.expand_dims(np_img, 0)
            if(self._use_float16 == "1"):
                img_batch = img_batch.astype(np.float16)
            else:
                img_batch = img_batch.astype(np.float32)

            img_batch /= 127.5
            img_batch -= -1.0
            self._interpreter.set_tensor(self._input_details[0]['index'], img_batch)
            start_time = time.time()
            self._interpreter.invoke()
            end_time = time.time()
            execution_time = end_time - start_time
            with open("/app/storage/time.log", "a") as f:
                           f.write(f"{execution_time}\n")
            print("Predictions: ")

    def _build(self):
        self._model_path = os.getenv('MODEL_PATH')
        self._ext_delegate = None
        if(os.getenv("TFLITE_DELEGATE") is not None):
            self._ext_delegate = [
            tflite.load_delegate(os.getenv("TFLITE_DELEGATE"))
            ]
        self._use_float16 = os.getenv("FLOAT_16")
        num_threads = os.getenv("NUM_THREAD")

        self._interpreter = tflite.Interpreter(
            model_path=self._model_path,
            experimental_delegates=self._ext_delegate,
            num_threads=num_threads
            )

        self._interpreter.allocate_tensors()
        self._input_details = self._interpreter.get_input_details()

