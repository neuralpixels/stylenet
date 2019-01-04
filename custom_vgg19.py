import tensorflow as tf
import numpy as np
from collections import OrderedDict
import os
import json
import time

# these weights have been trained over thousands of images. They are designed to be multiplied by the loss
# of each layer to normalize the loss differential from layer to layer

layer_weights = {
  "conv1_1": {
    "content": 0.0003927100042346865,
    "style": 0.27844879031181335
  },
  "conv1_2": {
    "content": 2.99037346849218e-05,
    "style": 0.0004943962558172643
  },
  "conv2_1": {
    "content": 2.0568952095345594e-05,
    "style": 0.0009304438135586679
  },
  "conv2_2": {
    "content": 1.073586827260442e-05,
    "style": 0.00040253016049973667
  },
  "conv3_1": {
    "content": 1.0999920050380751e-05,
    "style": 0.0001156232028733939
  },
  "conv3_2": {
    "content": 1.0808796105266083e-05,
    "style": 7.009495311649516e-05
  },
  "conv3_3": {
    "content": 4.947870365867857e-06,
    "style": 7.687774996156804e-06
  },
  "conv3_4": {
    "content": 1.2470403589759371e-06,
    "style": 8.033587732825254e-07
  },
  "conv4_1": {
    "content": 1.4441507119045127e-06,
    "style": 5.199814836487349e-07
  },
  "conv4_2": {
    "content": 2.3558966404380044e-06,
    "style": 2.2772749161958927e-06
  },
  "conv4_3": {
    "content": 5.842243808729108e-06,
    "style": 2.7995649361400865e-05
  },
  "conv4_4": {
    "content": 3.0219671316444874e-05,
    "style": 0.001985269133001566
  },
  "conv5_1": {
    "content": 6.438765558414161e-05,
    "style": 0.000784530770033598
  },
  "conv5_2": {
    "content": 0.00033032899955287576,
    "style": 0.018374426290392876
  },
  "conv5_3": {
    "content": 0.0016348531935364008,
    "style": 0.42564332485198975
  },
  "conv5_4": {
    "content": 0.02764303795993328,
    "style": 95.27446746826172
  }
}

_weights_vgg_style = {}


def _dequantize_weights(quantized_data, scale, min_val, original_dtype=np.float32):
    return quantized_data.astype(original_dtype) * scale + min_val


def _get_dtype(dtype_string, is_tf=False):
    if dtype_string == 'uint8':
        return tf.uint8 if is_tf else np.uint8
    elif dtype_string == 'uint16':
        return tf.uint16 if is_tf else np.uint16
    elif dtype_string == 'uint32':
        return tf.uint32 if is_tf else np.uint32
    elif dtype_string == 'uint64':
        return tf.uint64 if is_tf else np.uint64
    elif dtype_string == 'int16':
        return tf.int16 if is_tf else np.int16
    elif dtype_string == 'int32':
        return tf.int32 if is_tf else np.int32
    elif dtype_string == 'int64':
        return tf.int64 if is_tf else np.int64
    elif dtype_string == 'float16':
        return tf.float16 if is_tf else np.float16
    elif dtype_string == 'float32':
        return tf.float32 if is_tf else np.float32
    elif dtype_string == 'float64':
        return tf.float64 if is_tf else np.float64
    else:
        raise ValueError('Unknown dtype {}'.format(dtype_string))


def _load_weights(dtype='float32'):
    weights_folder = os.path.join(
        os.path.dirname(__file__), 'models/vgg19'
    )
    weight_dict = {}
    manifest_path = os.path.join(weights_folder, 'manifest.json')
    with open(manifest_path) as f:
        manifest = json.load(f)
        for weight_name, weight_obj in manifest['weights'].items():
            weight_file_path = os.path.join(weights_folder, weight_obj['filename'])

            with open(weight_file_path, "rb") as binary_file:
                # Read the whole file at once
                data = binary_file.read()
                if 'quantization' in weight_obj:
                    target_dtype = _get_dtype(weight_obj['dtype'])
                    quant = weight_obj['quantization']
                    weight_np_quant = np.frombuffer(data, dtype=_get_dtype(quant['dtype']))
                    weight_np = _dequantize_weights(
                        quantized_data=weight_np_quant,
                        scale=quant['scale'],
                        min_val=quant['min_value'],
                        original_dtype=target_dtype
                    )
                else:
                    weight_np = np.frombuffer(data, dtype=np.float32)
                weights_reshaped = np.reshape(weight_np, tuple(weight_obj['shape']))
                weights = tf.constant(
                    weights_reshaped, dtype=tf.float32, shape=weight_obj['shape'],
                    name='{}/{}'.format('vgg_style', weight_name)
                )
                if dtype == 'float16':
                    weights = tf.cast(weights, tf.float16)
                weight_dict[weight_name] = weights
    return weight_dict


def _process(input_tensor, network=None, reuse_weights=True, dtype='float32'):
    use_cudnn = tf.test.is_built_with_cuda()
    # data_format = (
    #     'channels_first' if tf.test.is_built_with_cuda() else 'channels_last')
    #
    # if data_format == 'channels_first':
    #     # Convert the inputs from channels_last (NHWC) to channels_first (NCHW).
    #     # This provides a large performance boost on GPU. See
    #     # https://www.tensorflow.org/performance/performance_guide#data_formats
    #     input_tensor = tf.transpose(input_tensor, [0, 3, 1, 2])

    layers = [
        'conv1_1', 'conv1_2', 'pool1',
        'conv2_1', 'conv2_2', 'pool2',
        'conv3_1', 'conv3_2', 'conv3_3', 'conv3_4', 'pool3',
        'conv4_1', 'conv4_2', 'conv4_3', 'conv4_4', 'pool4',
        'conv5_1', 'conv5_2', 'conv5_3', 'conv5_4', 'pool5'
    ]

    if network is None:
        network = OrderedDict()

    def _conv_layer(inputs, kernel_weights, bias_weights, name):
        with tf.name_scope(name=name):
            conv_out = tf.nn.conv2d(
                input=inputs,
                filter=kernel_weights,
                strides=(1, 1, 1, 1),
                padding='SAME',
                name='conv2d',
                use_cudnn_on_gpu=use_cudnn
            )
            bias_added = tf.nn.bias_add(conv_out, bias_weights, name='bias')
            return tf.nn.relu(bias_added, name='relu')

    def _pool_layer(inputs, name):
        with tf.name_scope(name=name):
            return tf.nn.max_pool(
                value=inputs,
                ksize=(1, 2, 2, 1),
                strides=(1, 2, 2, 1),
                padding='SAME',
                data_format='NHWC',
                name='max_pool'
            )

    def get_weights():
        global _weights_vgg_style
        nonlocal reuse_weights
        nonlocal dtype
        nonlocal input_tensor
        graph = input_tensor.graph
        if not reuse_weights or graph not in _weights_vgg_style:
            weights_vgg_style = _load_weights(dtype)
            if reuse_weights:
                _weights_vgg_style[graph] = weights_vgg_style
        else:
            weights_vgg_style = _weights_vgg_style[graph]
        return weights_vgg_style

    r, g, b = tf.split(axis=-1, num_or_size_splits=3, value=input_tensor)
    mean_pixel = [103.939, 116.779, 123.68]
    inputs = tf.concat(values=[b - mean_pixel[0], g - mean_pixel[1], r - mean_pixel[2]], axis=-1)

    network['vgg_input'] = inputs
    weights = get_weights()

    current = network['vgg_input']
    for name in layers:
        kind = name[:4]
        if kind == 'conv':
            kernels = weights['{}/kernel'.format(name)]
            bias = weights['{}/bias'.format(name)]
            current = _conv_layer(current, kernels, bias, name)
            network['{}'.format(name)] = current
        elif kind == 'pool':
            current = _pool_layer(current, name)
            network['{}'.format(name)] = current

    # return the network
    return network


class Vgg19(object):
    def avg_pool(self, bottom, name):
        with tf.name_scope(name=name):
            return tf.nn.avg_pool(
                value=bottom,
                ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1],
                padding='SAME',
                name='avg_pool',
                data_format='NHWC'
            )

    # Input should be an rgb image [batch, height, width, 3]
    # values scaled [0, 1]
    def build(self, rgb, train=False):
        start_time = time.time()
        print("build model started")
        rgb_scaled = rgb * 255.0
        net = _process(rgb_scaled)

        self.conv1_1 = net['conv1_1']
        self.conv1_2 = net['conv1_2']
        self.pool1 = net['pool1']

        self.conv2_1 = net['conv2_1']
        self.conv2_2 = net['conv2_2']
        self.pool2 = net['pool2']

        self.conv3_1 = net['conv3_1']
        self.conv3_2 = net['conv3_2']
        self.conv3_3 = net['conv3_3']
        self.conv3_4 = net['conv3_4']
        self.pool3 = net['pool3']

        self.conv4_1 = net['conv4_1']
        self.conv4_2 = net['conv4_2']
        self.conv4_3 = net['conv4_3']
        self.conv4_4 = net['conv4_4']
        self.pool4 = net['pool4']

        self.conv5_1 = net['conv5_1']
        self.conv5_2 = net['conv5_2']
        self.conv5_3 = net['conv5_3']
        self.conv5_4 = net['conv5_4']
        self.pool5 = net['pool5']

        self.data_dict = None
        print("build model finished: %ds" % (time.time() - start_time))
