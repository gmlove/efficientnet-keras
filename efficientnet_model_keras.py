import math
import re
from typing import List, Tuple, Union, Dict, TypeVar

import numpy as np
import keras
import tensorflow as tf


Tensor = TypeVar('Tensor')


def conv_kernel_initializer(shape, dtype=tf.float32):
    """Initialization for convolutional kernels.

    The main difference with tf.variance_scaling_initializer is that
    tf.variance_scaling_initializer uses a truncated normal with an uncorrected
    standard deviation, whereas here we use a normal distribution. Similarly,
    tf.contrib.layers.variance_scaling_initializer uses a truncated normal with
    a corrected standard deviation.

    Args:
      shape: shape of variable
      dtype: dtype of variable

    Returns:
      an initialization for the variable
    """
    kernel_height, kernel_width, _, out_filters = shape
    fan_out = int(kernel_height * kernel_width * out_filters)
    return tf.random_normal(
        shape, mean=0.0, stddev=np.sqrt(2.0 / fan_out), dtype=dtype)


def dense_kernel_initializer(shape, dtype=tf.float32):
    """Initialization for dense kernels.

    This initialization is equal to
      tf.variance_scaling_initializer(scale=1.0/3.0, mode='fan_out',
                                      distribution='uniform').
    It is written out explicitly here for clarity.

    Args:
      shape: shape of variable
      dtype: dtype of variable

    Returns:
      an initialization for the variable
    """
    init_range = 1.0 / np.sqrt(shape[1])
    return tf.random_uniform(shape, -init_range, init_range, dtype=dtype)


def drop_connect(inputs, is_training, drop_connect_rate):
    if not is_training:
      return inputs

    # Compute keep_prob
    # TODO(tanmingxing): add support for training progress.
    keep_prob = 1.0 - drop_connect_rate

    # Compute drop_connect tensor
    batch_size = tf.shape(inputs)[0]
    random_tensor = keep_prob
    random_tensor += tf.random_uniform([batch_size, 1, 1, 1], dtype=inputs.dtype)
    binary_tensor = tf.floor(random_tensor)
    output = tf.div(inputs, keep_prob) * binary_tensor
    return output


class EfficientNetGlobalParams:

    def __init__(self, batch_norm_momentum, batch_norm_epsilon, dropout_rate, data_format, num_classes,
                 width_coefficient, depth_coefficient, depth_divisor, min_depth, drop_connect_rate):
        self.batch_norm_momentum: float = batch_norm_momentum
        self.batch_norm_epsilon: float = batch_norm_epsilon
        self.dropout_rate: float = dropout_rate
        self.data_format: str = data_format
        self.num_classes: int = num_classes
        self.width_coefficient: float = width_coefficient
        self.depth_coefficient: float = depth_coefficient
        self.depth_divisor: int = depth_divisor
        self.min_depth: Union[int, None] = min_depth
        self.drop_connect_rate: float = drop_connect_rate

    @property
    def channel_first(self):
        return self.data_format == 'channels_first'

    @staticmethod
    def get(model_params: 'EfficientNetParams', num_classes=1000, drop_connect_rate=0.2) -> 'EfficientNetGlobalParams':
        return EfficientNetGlobalParams(
            batch_norm_momentum=0.99,
            batch_norm_epsilon=1e-3,
            dropout_rate=model_params.dropout_rate,
            drop_connect_rate=drop_connect_rate,
            data_format='channels_last',
            num_classes=num_classes,
            width_coefficient=model_params.width_coefficient,
            depth_coefficient=model_params.depth_coefficient,
            depth_divisor=8,
            min_depth=None)


class EfficientNetParams:

    def __init__(self, width_coefficient, depth_coefficient, resolution, dropout_rate):
        self.width_coefficient: float = width_coefficient
        self.depth_coefficient: float = depth_coefficient
        self.resolution: int = resolution
        self.dropout_rate: float = dropout_rate

    @staticmethod
    def for_model(model_name: str) -> 'EfficientNetParams':
        params_dict = {
            # (width_coefficient, depth_coefficient, resolution, dropout_rate)
            'efficientnet-b0': (1.0, 1.0, 224, 0.2),
            'efficientnet-b1': (1.0, 1.1, 240, 0.2),
            'efficientnet-b2': (1.1, 1.2, 260, 0.3),
            'efficientnet-b3': (1.2, 1.4, 300, 0.3),
            'efficientnet-b4': (1.4, 1.8, 380, 0.4),
            'efficientnet-b5': (1.6, 2.2, 456, 0.4),
            'efficientnet-b6': (1.8, 2.6, 528, 0.5),
            'efficientnet-b7': (2.0, 3.1, 600, 0.5),
        }
        return EfficientNetParams(*params_dict[model_name])


class EfficientNetBlockParams:

    def __init__(self, kernel_size: int, num_repeat: int, input_filters: int, output_filters: int, strides: List[int],
                 expand_ratio: int, id_skip: bool, se_ratio: Union[float, None]):
        self.kernel_size: int = kernel_size
        self.num_repeat: int = num_repeat
        self.input_filters: int = input_filters
        self.output_filters: int = output_filters
        self.expand_ratio: int = expand_ratio
        self.id_skip: bool = id_skip
        self.strides: List[int] = strides
        self.se_ratio: Union[float, None] = se_ratio

    def clone(self) -> 'EfficientNetBlockParams':
        return EfficientNetBlockParams(kernel_size=self.kernel_size,
                                       num_repeat=self.num_repeat,
                                       input_filters=self.input_filters,
                                       output_filters=self.output_filters,
                                       expand_ratio=self.expand_ratio,
                                       id_skip=self.id_skip,
                                       strides=self.strides,
                                       se_ratio=self.se_ratio)

    def update_filters(self, global_params: EfficientNetGlobalParams) -> 'EfficientNetBlockParams':
        params = self.clone()
        params.input_filters = EfficientNetBlockParams.round_filters_based_on_depth_multiplier(params.input_filters,
                                                                                               global_params)
        params.output_filters = EfficientNetBlockParams.round_filters_based_on_depth_multiplier(params.output_filters,
                                                                                                global_params)
        return params

    def update_repeats(self, global_params: EfficientNetGlobalParams):
        params = self.clone()
        params.num_repeat = EfficientNetBlockParams.round_repeats(params.num_repeat, global_params)
        return params

    def repeated_block_params(self) -> 'EfficientNetBlockParams':
        params = self.clone()
        params.input_filters = self.output_filters
        params.strides = [1, 1]
        return params

    @staticmethod
    def round_repeats(repeats: int, global_params: EfficientNetGlobalParams):
        """Round number of filters based on depth multiplier."""
        multiplier = global_params.depth_coefficient
        if not multiplier:
            return repeats
        return int(math.ceil(multiplier * repeats))

    @staticmethod
    def round_filters_based_on_depth_multiplier(filters, global_params: EfficientNetGlobalParams):
        orig_f = filters
        multiplier = global_params.width_coefficient
        divisor = global_params.depth_divisor
        min_depth = global_params.min_depth
        if not multiplier:
            return filters

        filters *= multiplier
        min_depth = min_depth or divisor
        new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
        # Make sure that round down does not go down by more than 10%.
        if new_filters < 0.9 * filters:
            new_filters += divisor
        tf.logging.info('round_filter input={} output={}'.format(orig_f, new_filters))
        return int(new_filters)

    @staticmethod
    def decode(encoded: str) -> 'EfficientNetBlockParams':
        ops = encoded.split('_')
        options = {}
        for op in ops:
            splits = re.split(r'(\d.*)', op)
            if len(splits) >= 2:
                key, value = splits[:2]
                options[key] = value

        if 's' not in options or len(options['s']) != 2:
            raise ValueError('Strides options should be a pair of integers.')

        return EfficientNetBlockParams(
            kernel_size=int(options['k']),
            num_repeat=int(options['r']),
            input_filters=int(options['i']),
            output_filters=int(options['o']),
            expand_ratio=int(options['e']),
            id_skip=('noskip' not in encoded),
            se_ratio=float(options['se']) if 'se' in options else None,
            strides=[int(options['s'][0]), int(options['s'][1])])

    @staticmethod
    def decode_list(encoded_list: List[str]) -> List['EfficientNetBlockParams']:
        params = []
        for encoded in encoded_list:
            params.append(EfficientNetBlockParams.decode(encoded))
        return params

    def encode(self) -> str:
        args = [
            'r%d' % self.num_repeat,
            'k%d' % self.kernel_size,
            's%d%d' % (self.strides[0], self.strides[1]),
            'e%s' % self.expand_ratio,
            'i%d' % self.input_filters,
            'o%d' % self.output_filters
        ]
        if 0 < self.se_ratio <= 1:
            args.append('se%s' % self.se_ratio)
        if self.id_skip is False:
            args.append('noskip')
        return '_'.join(args)

    @staticmethod
    def encode_list(block_params: List['EfficientNetBlockParams']) -> List[str]:
        return [block_param.encode() for block_param in block_params]

    @staticmethod
    def get_all() -> List['EfficientNetBlockParams']:
        blocks_args = [
            'r1_k3_s11_e1_i32_o16_se0.25', 'r2_k3_s22_e6_i16_o24_se0.25',
            'r2_k5_s22_e6_i24_o40_se0.25', 'r3_k3_s22_e6_i40_o80_se0.25',
            'r3_k5_s11_e6_i80_o112_se0.25', 'r4_k5_s22_e6_i112_o192_se0.25',
            'r1_k3_s11_e6_i192_o320_se0.25',
        ]
        return EfficientNetBlockParams.decode_list(blocks_args)


class MBConvBlock(object):
    """Mobile Inverted Residual Bottleneck."""

    def __init__(self, name: str, block_args: EfficientNetBlockParams, global_params: EfficientNetGlobalParams, drop_connect_rate):
        self.name = name
        self._block_args = block_args
        self._batch_norm_momentum = global_params.batch_norm_momentum
        self._batch_norm_epsilon = global_params.batch_norm_epsilon
        self._drop_connect_rate = drop_connect_rate
        if global_params.data_format == 'channels_first':
            self._channel_axis = 1
            self._spatial_dims = [2, 3]
        else:
            self._channel_axis = -1
            self._spatial_dims = [1, 2]
        self.has_se = (self._block_args.se_ratio is not None) \
                      and (self._block_args.se_ratio > 0) and (self._block_args.se_ratio <= 1)

        self._layers = {}
        self._build_layers()

    def block_args(self) -> EfficientNetBlockParams:
        return self._block_args

    def _layer_name(self, name: str) -> str:
        return '{}_{}'.format(self.name, name)

    def _build_layers(self):
        filters = self._block_args.input_filters * self._block_args.expand_ratio
        if self._block_args.expand_ratio != 1:
            # Expansion phase:
            self._expand_conv = keras.layers.Conv2D(
                name=self._layer_name('expand_conv'),
                filters=filters, kernel_size=[1, 1], strides=[1, 1], padding='same',
                use_bias=False, kernel_initializer=conv_kernel_initializer)
            self._bn0 = keras.layers.BatchNormalization(
                name=self._layer_name('expand_bn'),
                axis=self._channel_axis, momentum=self._batch_norm_momentum, epsilon=self._batch_norm_epsilon)
            self._act0 = Swish(name=self._layer_name('expand_act'))

        kernel_size = self._block_args.kernel_size
        # Depth-wise convolution phase:
        self._depthwise_conv = keras.layers.DepthwiseConv2D(
            name=self._layer_name('depthwise_conv'),
            kernel_size=[kernel_size, kernel_size], strides=self._block_args.strides, padding='same',
            use_bias=False, depthwise_initializer=conv_kernel_initializer)
        self._bn1 = keras.layers.BatchNormalization(
            name=self._layer_name('depthwise_bn'),
            axis=self._channel_axis, momentum=self._batch_norm_momentum, epsilon=self._batch_norm_epsilon)
        self._act1 = Swish(name=self._layer_name('depthwise_act'),)

        if self.has_se:
            num_reduced_filters = max(1, int(self._block_args.input_filters * self._block_args.se_ratio))
            # Squeeze and Excitation layer.
            self._se_mean = keras.layers.Lambda(
                name=self._layer_name('se_mean'),
                function=lambda x: tf.reduce_mean(x, self._spatial_dims, keep_dims=True))
            self._se_reduce = keras.layers.Conv2D(
                name=self._layer_name('se_reduce_conv'),
                filters=num_reduced_filters, kernel_size=[1, 1], strides=[1, 1], padding='same',
                use_bias=True, kernel_initializer=conv_kernel_initializer)
            self._se_act = Swish(name=self._layer_name('se_act'))
            self._se_expand = keras.layers.Conv2D(
                name=self._layer_name('se_expand_conv'),
                filters=filters, kernel_size=[1, 1], strides=[1, 1], padding='same',
                use_bias=True, kernel_initializer=conv_kernel_initializer)
            self._se_apply = keras.layers.Lambda(
                name=self._layer_name('se_apply'),
                function=lambda se_tensor_and_input_tensor: tf.sigmoid(se_tensor_and_input_tensor[0]) * se_tensor_and_input_tensor[1])

        # Output phase:
        filters = self._block_args.output_filters
        self._project_conv = keras.layers.Conv2D(
            name=self._layer_name('project_conv'),
            filters=filters, kernel_size=[1, 1], strides=[1, 1], padding='same',
            use_bias=False, kernel_initializer=conv_kernel_initializer)
        self._bn2 = keras.layers.BatchNormalization(
            name=self._layer_name('project_bn'),
            axis=self._channel_axis, momentum=self._batch_norm_momentum, epsilon=self._batch_norm_epsilon)
        self._drop_connect = DropConnect(self._drop_connect_rate, name=self._layer_name('project_drop_connect')) \
            if self._drop_connect_rate else None
        self._add = keras.layers.Add(name=self._layer_name('project_add'))

    def __setattr__(self, name, value):
        if isinstance(value, (keras.layers.Layer, keras.engine.network.Network)):
            if value not in self._layers:
                self._layers[name] = value
        super(MBConvBlock, self).__setattr__(name, value)

    @property
    def layers(self) -> Dict[str, Union[keras.layers.Layer, keras.engine.network.Network]]:
        return self._layers.copy()

    def _connect_se(self, input_tensor: Tensor) -> Tensor:
        se_tensor = self._se_mean(input_tensor)
        se_tensor = self._se_expand(self._se_act(self._se_reduce(se_tensor)))
        tf.logging.info('Built Squeeze and Excitation with tensor shape: %s' % se_tensor.shape)
        return self._se_apply([se_tensor, input_tensor])

    def connect_layers(self, inputs: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        tf.logging.info('Block input: %s shape: %s' % (inputs.name, inputs.shape))
        if self._block_args.expand_ratio != 1:
            x = self._act0(self._bn0(self._expand_conv(inputs)))
        else:
            x = inputs
        tf.logging.info('Expand: %s shape: %s' % (x.name, x.shape))

        x = self._act1(self._bn1(self._depthwise_conv(x)))
        tf.logging.info('DWConv: %s shape: %s' % (x.name, x.shape))

        if self.has_se:
            with tf.variable_scope('se'):
                x = self._connect_se(x)

        endpoints = {'expansion_output': x}

        x = self._bn2(self._project_conv(x))
        if self._block_args.id_skip:
            if all(s == 1 for s in self._block_args.strides) and self._block_args.input_filters == self._block_args.output_filters:
                if self._drop_connect:
                    x = self._drop_connect(x)
                x = self._add([x, inputs])
        tf.logging.info('Project: %s shape: %s' % (x.name, x.shape))
        return x, endpoints


class EfficientNetModel(object):

    def __init__(self, model_name, input_shape=(None, None, 3), num_classes=1000):
        self._global_params: EfficientNetGlobalParams = EfficientNetGlobalParams.get(
            EfficientNetParams.for_model(model_name), num_classes=num_classes)
        self._blocks_args: List[EfficientNetBlockParams] = EfficientNetBlockParams.get_all()
        self._build_layers()

        inputs = keras.Input(input_shape, name='data')
        outputs, endpoints = self._connect_layers(inputs)

        self.endpoints = endpoints
        self.model = keras.models.Model(inputs, outputs, name=model_name)

    def _build_layers(self):
        # Stem part.
        batch_norm_momentum = self._global_params.batch_norm_momentum
        batch_norm_epsilon = self._global_params.batch_norm_epsilon
        channel_axis = 1 if self._global_params.channel_first else -1
        self._conv_stem = keras.layers.Conv2D(
            name='stem_conv',
            filters=EfficientNetBlockParams.round_filters_based_on_depth_multiplier(32, self._global_params),
            kernel_size=[3, 3], strides=[2, 2], padding='same',
            use_bias=False, kernel_initializer=conv_kernel_initializer)
        self._bn0 = keras.layers.BatchNormalization(name='stem_bn',
                                                    axis=channel_axis, momentum=batch_norm_momentum, epsilon=batch_norm_epsilon)
        self._act0 = Swish(name='stem_act')

        # Blocks
        blocks_count = sum([arg.num_repeat for arg in self._blocks_args])
        self._blocks = []
        for i, block_args in enumerate(self._blocks_args):
            assert block_args.num_repeat > 0
            block_args = block_args.update_filters(self._global_params).update_repeats(self._global_params)
            self._blocks.append(MBConvBlock('block_{}_{}'.format(i + 1, 1), block_args, self._global_params,
                                            drop_connect_rate=self._block_drop_rate(len(self._blocks), blocks_count)))

            repeated_block_args = block_args.repeated_block_params()
            for block_repeat_idx in range(block_args.num_repeat - 1):
                block_name = 'block_{}_{}'.format(i + 1, block_repeat_idx + 2)
                self._blocks.append(MBConvBlock(block_name, repeated_block_args, self._global_params,
                                                drop_connect_rate=self._block_drop_rate(len(self._blocks), blocks_count)))

        # Head part.
        self._conv_head = keras.layers.Conv2D(
            name='head_conv',
            filters=EfficientNetBlockParams.round_filters_based_on_depth_multiplier(1280, self._global_params),
            kernel_size=[1, 1], strides=[1, 1], padding='same',
            use_bias=False, kernel_initializer=conv_kernel_initializer)
        self._bn1 = keras.layers.BatchNormalization(
            name='head_bn', axis=channel_axis, momentum=batch_norm_momentum, epsilon=batch_norm_epsilon)
        self._act1 = Swish(name='head_act')

        self._avg_pooling = keras.layers.GlobalAveragePooling2D(name='head_pooling', data_format=self._global_params.data_format)
        self._fc = keras.layers.Dense(name='head_dense', units=self._global_params.num_classes,
                                      kernel_initializer=dense_kernel_initializer)

        self._dropout = keras.layers.Dropout(name='head_dropout', rate=self._global_params.dropout_rate) \
            if self._global_params.dropout_rate > 0 else None

    def _block_drop_rate(self, block_idx, blocks_count):
        drop_rate = self._global_params.drop_connect_rate
        if drop_rate:
            drop_rate *= float(block_idx) / blocks_count
            tf.logging.info('block_%s drop_connect_rate: %s' % (block_idx, drop_rate))
        return drop_rate

    def _connect_layers(self, inputs: keras.Input) -> Tuple[Tensor, Dict[str, Tensor]]:
        endpoints = {}
        # Calls Stem layers
        with tf.variable_scope('stem'):
            outputs = self._act0(self._bn0(self._conv_stem(inputs)))
        tf.logging.info('Built stem layers with output shape: %s' % outputs.shape)
        endpoints['stem'] = outputs

        # Calls blocks.
        reduction_idx = 0
        for idx, block in enumerate(self._blocks):
            is_reduction = False
            if (idx == len(self._blocks) - 1) or self._blocks[idx + 1].block_args().strides[0] > 1:
                is_reduction = True
                reduction_idx += 1

            with tf.variable_scope('blocks_%s' % idx):
                outputs, block_endpoints = block.connect_layers(outputs)
                endpoints['block_%s' % idx] = outputs
                if is_reduction:
                    endpoints['reduction_%s' % reduction_idx] = outputs
                for k, v in block_endpoints.items():
                    endpoints['block_%s/%s' % (idx, k)] = v
                    if is_reduction:
                        endpoints['reduction_%s/%s' % (reduction_idx, k)] = v
        endpoints['global_pool'] = outputs

        # Calls final layers and returns logits.
        with tf.variable_scope('head'):
            outputs = self._act1(self._bn1(self._conv_head(outputs)))
            outputs = self._avg_pooling(outputs)
            if self._dropout:
                outputs = self._dropout(outputs)
            outputs = self._fc(outputs)
            endpoints['head'] = outputs
        return outputs, endpoints


class EfficientNetModelBuilder(object):

    def build(self, model_name, input_shape=(None, None, 3), num_classes=1000) -> keras.Model:
        return EfficientNetModel(model_name, input_shape, num_classes).model


class Swish(keras.layers.Layer):

    def __init__(self, **kwargs):
        super(Swish, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        return tf.nn.swish(inputs)

    def get_config(self):
        config = {}
        base_config = super(Swish, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


class DropConnect(keras.layers.Layer):

    def __init__(self, drop_connect_rate, **kwargs):
        super(DropConnect, self).__init__(**kwargs)
        self.drop_connect_rate = drop_connect_rate

    def call(self, inputs, training=None):
        return drop_connect(inputs, training, self.drop_connect_rate)

    def get_config(self):
        config = {'drop_connect_rate': self.drop_connect_rate}
        base_config = super(DropConnect, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape
