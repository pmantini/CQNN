import tensorflow as tf
from tensorflow.keras.layers import Layer, InputSpec
import tensorflow.keras.backend as K

class QConv2d(Layer):

    def __init__(self, n_clusters, kernel_size=3, stride=1, padding = "SAME", is_neg=1, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(QConv2d, self).__init__(**kwargs)
        self.n_clusters = n_clusters

        self.input_spec = InputSpec(ndim=4)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.is_neg = is_neg

    def build(self, input_shape):
        assert len(input_shape) == 4
        input_dim = input_shape[1]

        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim, input_dim, input_shape[3]))
        channels = K.variable(input_shape[3], dtype="int32")
        self.sqaure_dim = self.kernel_size*self.kernel_size*K.get_value(channels)+1
        self.quadratic = self.add_weight("quad", (self.n_clusters, self.sqaure_dim, self.sqaure_dim), initializer='glorot_uniform')

        self.built = True

    def call(self, inputs, **kwargs):

        pathces = tf.image.extract_patches(inputs,  [1,self.kernel_size, self.kernel_size, 1], [1, self.stride,
                                                                              self.stride,1], [1,1,1,1], self.padding)

        padding = tf.constant([[0, 0], [0, 0], [0, 0], [0, 1]])
        patches_expanded = tf.pad(pathces, padding, "CONSTANT", constant_values=1)

        q = K.sum(K.dot(patches_expanded, self.quadratic) * K.expand_dims(patches_expanded, axis=3), axis=4)
        if self.is_neg:
            return -q
        else:
            return q



    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 4
        if self.padding == "SAME":
            p = self.kernel_size//2
        else:
            p = 0
        return None, (input_shape[1]-self.kernel_size + 2*p)/self.stride+1, (input_shape[2]-self.kernel_size + 2*p)/self.stride+1, self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(QConv2d, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

