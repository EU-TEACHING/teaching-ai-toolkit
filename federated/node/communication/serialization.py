from tensorflow import keras
import numpy as np


def model_to_packet_body(model):
    config = model.to_json()
    weights = [w.tolist() for w in model.get_weights()]
    return {'config': config, 'weights': weights}

def model_from_packet_body(model_json):
    model = keras.models.model_from_json(model_json['config'])
    model.set_weights([np.array(w) for w in model_json['weights']])
    return model
