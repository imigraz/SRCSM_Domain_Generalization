import random
import tensorflow as tf
from tensorflow.keras.layers import Conv3D, AveragePooling3D, AlphaDropout, Activation, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow_train_v2.networks.unet_base import UnetBase
from tensorflow_train_v2.layers.layers import Sequential, ConcatChannels, UpSampling3DLinear


def frobenius_norm(matrix):
    matrix_cast = tf.cast(matrix, dtype=tf.float32)
    norm = tf.sqrt(tf.reduce_sum(tf.square(matrix_cast)))
    ret_norm = tf.cast(norm, dtype=tf.float16)
    return ret_norm if ret_norm != 0 else tf.cast(0.00001, dtype=tf.float16)


class UnetAvgLinear3D(UnetBase):
    """
    U-Net with average pooling and linear upsampling.
    """
    def __init__(self,
                 num_filters_base,
                 repeats=2,
                 dropout_ratio=0.0,
                 kernel_size=None,
                 activation=tf.nn.relu,
                 kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_in', distribution='truncated_normal'),
                 alpha_dropout=False,
                 data_format='channels_first',
                 padding='same',
                 *args, **kwargs):
        super(UnetAvgLinear3D, self).__init__(*args, **kwargs)
        self.num_filters_base = num_filters_base
        self.repeats = repeats
        self.dropout_ratio = dropout_ratio
        self.kernel_size = kernel_size or [3] * 3
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.alpha_dropout = alpha_dropout
        self.data_format = data_format
        self.padding = padding
        self.init_layers()

    def downsample(self, current_level):
        """
        Create and return downsample keras layer for the current level.
        :param current_level: The current level.
        :return: The keras.layers.Layer.
        """
        return AveragePooling3D([2] * 3, data_format=self.data_format)

    def upsample(self, current_level):
        """
        Create and return upsample keras layer for the current level.
        :param current_level: The current level.
        :return: The keras.layers.Layer.
        """
        return UpSampling3DLinear([2] * 3, data_format=self.data_format)

    def combine(self, current_level):
        """
        Create and return combine keras layer for the current level.
        :param current_level: The current level.
        :return: The keras.layers.Layer.
        """
        return ConcatChannels(data_format=self.data_format)

    def contracting_block(self, current_level):
        """
        Create and return the contracting block keras layer for the current level.
        :param current_level: The current level.
        :return: The keras.layers.Layer.
        """
        layers = []
        for i in range(self.repeats):
            layers.append(self.conv(current_level, str(i)))
            if self.dropout_ratio > 0:
                if self.alpha_dropout:
                    layers.append(AlphaDropout(self.dropout_ratio))
                else:
                    layers.append(Dropout(self.dropout_ratio))
        return Sequential(layers, name='contracting' + str(current_level))

    def expanding_block(self, current_level):
        """
        Create and return the expanding block keras layer for the current level.
        :param current_level: The current level.
        :return: The keras.layers.Layer.
        """
        layers = []
        for i in range(self.repeats):
            layers.append(self.conv(current_level, str(i)))
            if self.dropout_ratio > 0:
                if self.alpha_dropout:
                    layers.append(AlphaDropout(self.dropout_ratio))
                else:
                    layers.append(Dropout(self.dropout_ratio))
        return Sequential(layers, name='expanding' + str(current_level))

    def conv(self, current_level, postfix):
        """
        Create and return a convolution layer for the current level with the current postfix.
        :param current_level: The current level.
        :param postfix:
        :return:
        """
        return Conv3D(self.num_filters_base,
                      self.kernel_size,
                      name='conv' + postfix,
                      activation=self.activation,
                      data_format=self.data_format,
                      kernel_initializer=self.kernel_initializer,
                      kernel_regularizer=l2(1.0),
                      padding=self.padding)


def activation_fn_output_kernel_initializer(activation):
    if activation == 'none':
        activation_fn = None
        kernel_initializer = tf.keras.initializers.TruncatedNormal(stddev=0.001)
    elif activation == 'tanh':
        activation_fn = tf.tanh
        kernel_initializer = tf.keras.initializers.TruncatedNormal(stddev=0.0001)
    elif activation == 'abs_tanh':
        activation_fn = lambda x, *args, **kwargs: tf.abs(tf.tanh(x))
        kernel_initializer = tf.keras.initializers.TruncatedNormal(stddev=0.0001)
    elif activation == 'square_tanh':
        activation_fn = lambda x: tf.tanh(x * x)
        kernel_initializer = tf.keras.initializers.TruncatedNormal(stddev=0.05)
    elif activation == 'inv_gauss':
        activation_fn = lambda x: 1.0 - tf.math.exp(-tf.square(x))
        kernel_initializer = tf.keras.initializers.TruncatedNormal(stddev=0.05)
    elif activation == 'squash':
        a = 5
        b = 1
        l = 1
        activation_fn = lambda x: 1.0 / (l * b) * tf.math.log((1.0 + tf.math.exp(b * (x - (a - l / 2.0)))) / (1.0 + tf.math.exp(b * (x - (a + l / 2.0)))))
        kernel_initializer = tf.keras.initializers.TruncatedNormal(stddev=0.05)
    elif activation == 'sigmoid':
        activation_fn = tf.nn.sigmoid
        kernel_initializer = tf.keras.initializers.TruncatedNormal(stddev=0.05)
    return activation_fn, kernel_initializer


class Unet(tf.keras.Model):
    def __init__(self, num_labels, num_filters_base=64, num_levels=4, activation='relu', data_format='channels_first', padding='same', dropout_ratio=0.0, **kwargs):
        super(Unet, self).__init__()
        self.data_format = data_format
        num_filters_base = num_filters_base
        if activation == 'relu':
            activation_fn = tf.nn.relu
        elif activation == 'lrelu':
            activation_fn = lambda x: tf.nn.leaky_relu(x, alpha=0.1)
        self.unet = UnetAvgLinear3D(num_filters_base=num_filters_base, num_levels=num_levels, kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_in', distribution='truncated_normal'), activation=activation_fn, dropout_ratio=dropout_ratio, data_format=data_format, padding=padding)
        self.prediction = Sequential([Conv3D(num_labels, [1] * 3, name='prediction', kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_in', distribution='truncated_normal'), activation=None, data_format=data_format, padding=padding),
                                    Activation(None, dtype='float32', name='prediction')])
        self.single_output = num_labels == 1

    def call(self, inputs, training, **kwargs):
        node = self.unet(inputs, training=training)
        prediction = self.prediction(node, training=training)
        if self.single_output:
            return prediction
        else:
            return prediction, node, prediction, prediction, prediction


class RCNet(tf.keras.Model):
    def __init__(self, num_filters_last_layer=1, num_filters_base=2, data_format='channels_first', padding='same', **kwargs):
        super().__init__()
        self.data_format = data_format
        activation_fn = lambda x: tf.nn.leaky_relu(x, alpha=0.1)
        self.alpha = random.uniform(0, 1)
        num_layers = 4
        kernel_candidates = [1, 3]
        self.layer_list = []
        for i in range(num_layers):
            kernel_size = random.choice(kernel_candidates)
            cur_num_filters = num_filters_last_layer if i == num_layers - 1 else num_filters_base
            cur_activation = None if i == num_layers - 1 else activation_fn
            self.layer_list.append(Conv3D(cur_num_filters, [kernel_size] * 3, name=f'conv{i+1}', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0), activation=cur_activation, data_format=data_format, padding=padding))
        self.prediction = Sequential(self.layer_list + [Activation(activation_fn, dtype='float16', name='prediction')])

    def call(self, inputs, training=True, **kwargs):
        outputs = self.prediction(inputs, training=training)
        prediction = outputs * self.alpha + inputs * (1 - self.alpha)
        prediction_norm = prediction / frobenius_norm(prediction) * frobenius_norm(inputs)
        return prediction_norm

