import os 
import tensorflow as tf
from tensorflow import keras
import numpy as np

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
                x = keras.layers.RNN(
                    ReservoirCell(
                        units=int(os.environ['UNITS']),
                        leaky=float(os.environ['LEAKY']),
                        spectral_radius=float(os.environ['RHO']),
                        connectivity_input=float(os.environ['CONNECTIVITY'])
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


class ReservoirCell(keras.layers.AbstractRNNCell):
    """
    Implementation of a shallow reservoir to be used as cell of a Recurrent Neural Network
    
    Args:
    units: the number of recurrent neurons in the reservoir
    input_scaling: the max abs value of a weight in the input-reservoir connections
                    note that whis value also scales the unitary input bias 
    spectral_radius: the max abs eigenvalue of the recurrent weight matrix
    leaky: the leaking rate constant of the reservoir
    connectivity_input: number of outgoing connections from each input unit to the reservoir
    connectivity_recurrent: number of incoming recurrent connections for each reservoir unit
    """
    
    def __init__(self,
                 units: int,
                 input_scaling: float = 1.,
                 spectral_radius: float = 0.99,
                 leaky: float = 1., 
                 connectivity_input: int = 10, 
                 connectivity_recurrent: int = 10,
                 **kwargs):
        
        self.units = units
        self.state_size = units
        self.input_scaling = input_scaling
        self.spectral_radius = spectral_radius
        self.leaky = leaky
        self.connectivity_input = connectivity_input
        self.connectivity_recurrent = connectivity_recurrent
        super().__init__(**kwargs)
        
    def build(self, input_shape):
        
        self.W_in = self.add_weight(
            "W_in", 
            shape=(input_shape[-1], self.units),
            initializer=lambda: sparse_tensor(input_shape[-1], self.units, self.connectivity_input) * self.input_scaling,
            trainable=False
        )
        
        W = sparse_tensor(self.units, self.units, C=self.connectivity_recurrent)

        if self.leaky == 1:
            e,_ = tf.linalg.eig(tf.sparse.to_dense(W))
            rho = max(abs(e))
            W = W * (self.spectral_radius / rho)
            W_hat = W
        else:
            I = sparse_eye(self.units)
            W2 = tf.sparse.add(I * (1-self.leaky), W * self.leaky)
            e,_ = tf.linalg.eig(tf.sparse.to_dense(W2))
            rho = max(abs(e))
            W2 = W2 * (self.spectral_radius / rho)
            W_hat =  tf.sparse.add(W2, I * (self.leaky - 1)) * (1/self.leaky) 

        self.W_hat = self.add_weight(
            "W_hat",
            shape=(self.units, self.units),
            initializer=lambda: W_hat,
            trainable=False
        )
        self.b = self.add_weight(
            "b",
            shape=(self.units),
            initializer=lambda: tf.random.uniform(shape = (self.units,), minval = -1, maxval = 1) * self.input_scaling,
            trainable=False
        )
        
        self.leaky = self.add_weight(
            "leaky",
            shape=(),
            initializer=lambda: tf.tensor(self.leaky),
            trainable=False
        )

        self.built = True

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return tf.zeros((batch_size, self.units))

    def call(self, inputs, states):
        #computes the output of the cell givne the input and previous state
        prev_output = states[0]

        in_signal = tf.sparse.sparse_dense_matmul(inputs, self.W_in) \
                    + tf.sparse.sparse_dense_matmul(prev_output, self.W_hat)
        output = (1-self.leaky)*prev_output + self.leaky * tf.nn.tanh(in_signal + self.b)
        
        return output, [output]


def sparse_eye(M):
    dense_shape = (M,M)
    indices = np.zeros((M,2))
    for i in range(M):
        indices[i,:] = [i,i]
    values = np.ones(shape = (M,)).astype('f')

    W = tf.sparse.reorder(tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape))
    return W

def sparse_tensor(M, N, C=1):
    dense_shape = (M,N) #the shape of the dense version of the matrix

    indices = np.zeros((M * C,2)) #indices of non-zero elements initialization
    k = 0
    for i in range(M):
        #the indices of non-zero elements in the i-th row of the matrix
        idx =np.random.choice(N, size=C, replace=False)
        for j in range(C):
            indices[k,:] = [i,idx[j]] if M != N else [idx[j],i]
            k = k + 1
    values = 2*(2*np.random.rand(M*C).astype('f')-1)
    W = tf.sparse.reorder(tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape))
    return W