from hyperopt import hp
import numpy as np
from hyperopt.pyll.base import scope

model_spaces = {
    "lstm-ae": {
        'learning_rate': hp.loguniform('learning_rate', np.log(0.001), np.log(0.005)),
        'units': scope.int(hp.quniform('units', 80, 100, 6)),
        'batch_size': scope.int(hp.quniform('batch_size', 32, 64, 25)),
        'epochs': scope.int(hp.quniform('epochs', 50, 150, 50))
    }
}