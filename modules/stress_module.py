import os

import tensorflow as tf
import tensorflow_addons as tfa

from base.node import TEACHINGNode
from base.communication.packet import DataPacket
from .base_module import LearningModule


class StressModule(LearningModule):

    def __init__(self):
        super(StressModule, self).__init__()
        self._model_path = os.getenv('MODEL_PATH')
        self._build()
    
    @TEACHINGNode(produce=True, consume=True)
    def __call__(self, input_fn):

        for msg in input_fn:
            x = tf.constant([[[msg.body['eda']]]])
            stress_value = self._model(x)
            yield DataPacket(
                topic='prediction.stress.value', 
                timestamp=msg.timestamp,
                body={'stress': tf.squeeze(stress_value).numpy()})

    def _build(self):
        if self._model_path is not None:
            self._model = tf.keras.models.load_model(self._model_path)
        else:
            inputs = tf.keras.Input(batch_shape=(1, 1, int(os.environ['INPUT_SIZE'])))
            for i in range(int(os.environ['LAYERS'])):
                x = tfa.layers.ESN(
                    units=int(os.environ['UNITS']),
                    connectivity=float(os.environ['CONNECTIVITY']),
                    leaky=float(os.environ['LEAKY']),
                    spectral_radius=float(os.environ['RHO']),
                    kernel_initializer='glorot_uniform',
                    recurrent_initializer=SparseRecurrentTensor(
                        int(os.environ['UNITS']),
                        float(os.environ['LEAKY']),
                        float(os.environ['RHO'])
                    ),
                    return_sequences=True,
                    stateful=True
                )(inputs if i == 0 else x)
            outputs = tf.keras.layers.Dense(
                int(os.environ['N_CLASSES']), 
                activation=('sigmoid' if int(os.environ['N_CLASSES']) <= 2 else 'softmax')
            )(x)
            self._model = tf.keras.Model(inputs=inputs, outputs=outputs, name="stress_model")

        self._model.summary()


class SparseRecurrentTensor(tf.keras.initializers.Initializer):

  def __init__(self, M, leaky, spectral_radius, connectivity=10, W=None):
    self.M = M
    self.leaky = leaky
    self.spectral_radius = spectral_radius
    self.connectivity = connectivity
    self.W = W

  def __call__(self, shape, dtype=None, **kwargs):
    if not self.W:
      initializer = tf.keras.initializers.GlorotUniform()
      recurrent_weights = initializer(shape, dtype)
      connectivity_mask = tf.cast(tf.math.less_equal(tf.random.uniform(shape), self.connectivity), dtype)
      recurrent_weights = tf.math.multiply(recurrent_weights, connectivity_mask)
      abs_eig_values = tf.abs(tf.linalg.eig(recurrent_weights)[0])
      scaling_factor = tf.math.divide_no_nan(
          self.spectral_radius, tf.reduce_max(abs_eig_values)
      )
      self.W = tf.multiply(recurrent_weights, scaling_factor)
    return self.W

  def get_config(self):
    config = super().get_config().copy()
    config.update({"M": self.M, "leaky": self.leaky, "spectral_radius": self.spectral_radius,
        "connectivity": self.connectivity, "W": self.W.numpy()
    })
    return config
