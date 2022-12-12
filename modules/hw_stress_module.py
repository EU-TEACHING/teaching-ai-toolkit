import os, pickle, time
import pandas as pd

from base.node import TEACHINGNode
from base.communication.packet import DataPacket
from .base_module import LearningModule


class HWStressModule(LearningModule):

    FED_TOPIC = 'hwstressmodule'
    
    @TEACHINGNode(produce=True, consume=False)
    def __call__(self):
        while True:
            last_mod = os.stat(self._file_path).st_mtime
            if last_mod > self._mod_time:
                df = pd.read_csv(
                    self._file_path, 
                    skiprows=lambda i: i >= self._last_idx
                )

                values = df.drop('TIMESTAMP', axis=1)
                values['STRESSED'] = self.model.predict(values[self._predictors].values).tolist()
                yield DataPacket(
                    topic='prediction.hwstress.value', 
                    timestamp=df['TIMESTAMP'].tolist(),
                    body=values.to_dict(orient='records'))
            else:
                time.sleep(1)

    def _build(self):
        super(HWStressModule, self)._build()
        if self._model_path is not None and os.path.exists(self._model_path):
            with open(self._model_path, 'rb') as f:
                mdl = pickle.load(f)
                self._predictors = mdl['headers']
                self._model = mdl['model']
        else:
            raise FileNotFoundError("Invalid path for the given model.")
        
        self._file_path = os.environ['FILE_PATH']

        self._mod_time = -1
        self._last_idx = 0
    