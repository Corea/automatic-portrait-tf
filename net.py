import numpy as np
import tensorflow as tf


class FCN8s:
    net = {}
    output_dim = 0

    def __init__(self, output_dim):
        self.output_dim = output_dim

        self.net['image'] = tf.placeholder(tf.float32,
                                           shape=[None, None, None, 3])
        self.net['input'] = tf.pad(self.net['image'],
                                   [[0, 0], [99, 99], [99, 99], [0, 0]])
        self.net['drop_rate'] = tf.placeholder(tf.float32)

        self.build_layer('conv', 'conv1_1', 'input', shape=[3, 3, 3, 64])
        self.build_layer('relu', 'relu1_1', 'conv1_1')
        self.build_layer('conv', 'conv1_2', 'relu1_1', shape=[3, 3, 64, 64])
        self.build_layer('relu', 'relu1_2', 'conv1_2')
        self.build_layer('pool', 'pool1', 'relu1_2')

        self.build_layer('conv', 'conv2_1', 'pool1', shape=[3, 3, 64, 128])
        self.build_layer('relu', 'relu2_1', 'conv2_1')
        self.build_layer('conv', 'conv2_2', 'relu2_1', shape=[3, 3, 128, 128])
        self.build_layer('relu', 'relu2_2', 'conv2_2')
        self.build_layer('pool', 'pool2', 'relu2_2')

        self.build_layer('conv', 'conv3_1', 'pool2', shape=[3, 3, 128, 256])
        self.build_layer('relu', 'relu3_1', 'conv3_1')
        self.build_layer('conv', 'conv3_2', 'relu3_1', shape=[3, 3, 256, 256])
        self.build_layer('relu', 'relu3_2', 'conv3_2')
        self.build_layer('conv', 'conv3_3', 'relu3_2', shape=[3, 3, 256, 256])
        self.build_layer('relu', 'relu3_3', 'conv3_3')
        self.build_layer('pool', 'pool3', 'relu3_3')

        self.build_layer('conv', 'conv4_1', 'pool3', shape=[3, 3, 256, 512])
        self.build_layer('relu', 'relu4_1', 'conv4_1')
        self.build_layer('conv', 'conv4_2', 'relu4_1', shape=[3, 3, 512, 512])
        self.build_layer('relu', 'relu4_2', 'conv4_2')
        self.build_layer('conv', 'conv4_3', 'relu4_2', shape=[3, 3, 512, 512])
        self.build_layer('relu', 'relu4_3', 'conv4_3')
        self.build_layer('pool', 'pool4', 'relu4_3')

        self.build_layer('conv', 'conv5_1', 'pool4', shape=[3, 3, 512, 512])
        self.build_layer('relu', 'relu5_1', 'conv5_1')
        self.build_layer('conv', 'conv5_2', 'relu5_1', shape=[3, 3, 512, 512])
        self.build_layer('relu', 'relu5_2', 'conv5_2')
        self.build_layer('conv', 'conv5_3', 'relu5_2', shape=[3, 3, 512, 512])
        self.build_layer('relu', 'relu5_3', 'conv5_3')
        self.build_layer('pool', 'pool5', 'relu5_3')

        self.build_layer('conv', 'fc6', 'pool5', shape=[7, 7, 512, 4096],
                         padding='VALID')
        self.build_layer('relu', 'relu6', 'fc6')
        self.build_layer('drop', 'drop6', 'relu6', drop_layer_name='drop_rate')

        self.build_layer('conv', 'fc7', 'drop6', shape=[1, 1, 4096, 4096])
        self.build_layer('relu', 'relu7', 'fc7')
        self.build_layer('drop', 'drop7', 'relu7', drop_layer_name='drop_rate')
        self.build_layer('conv', 'score_fr', 'drop7',
                         shape=[1, 1, 4096, output_dim])

        self.build_layer('deconv', 'upscore2', 'score_fr',
                         shape=[4, 4, output_dim, output_dim],
                         strides=[1, 2, 2, 1])
        self.build_layer('conv', 'score_pool4', 'pool4',
                         shape=[1, 1, 512, output_dim])
        self.build_layer('slice', 'score_pool4c', 'score_pool4',
                         begin=[0, 5, 5, 0], shape_layer_name='upscore2')
        self.net['fuse_pool4'] = self.net['upscore2'] + \
            self.net['score_pool4c']

        self.build_layer('deconv', 'upscore_pool4', 'fuse_pool4',
                         shape=[4, 4, output_dim, output_dim],
                         strides=[1, 2, 2, 1])
        self.build_layer('conv', 'score_pool3', 'pool3',
                         shape=[1, 1, 256, output_dim])
        self.build_layer('slice', 'score_pool3c', 'score_pool3',
                         begin=[0, 9, 9, 0], shape_layer_name='upscore_pool4')
        self.net['fuse_pool3'] = self.net['upscore_pool4'] + \
            self.net['score_pool3c']

        self.build_layer('deconv', 'upscore8', 'fuse_pool3',
                         shape=[16, 16, output_dim, output_dim],
                         strides=[1, 8, 8, 1])

        b = tf.shape(self.net['image'])[0]
        h = tf.shape(self.net['image'])[1]
        w = tf.shape(self.net['image'])[2]
        d = tf.shape(self.net['upscore8'])[3]
        output_shape = tf.pack([b, h, w, d])
        self.net['score'] = tf.slice(self.net['upscore8'], [0, 31, 31, 0],
                                     output_shape)

    def build_conv(self, layer_name, input_layer, shape, strides=[1, 1, 1, 1],
                   padding='SAME'):
        bias_shape = shape[-1:]

        weight_name = layer_name + '_weight'
        bias_name = layer_name + '_bias'
        weight = tf.get_variable(weight_name, shape=shape)
        bias = tf.get_variable(bias_name, shape=bias_shape)
        self.net[weight_name] = weight
        self.net[bias_name] = bias

        self.net[layer_name] = tf.nn.bias_add(
            tf.nn.conv2d(input_layer, weight, strides=strides, padding=padding,
                         name=layer_name), bias)

    def build_relu(self, layer_name, input_layer):
        self.net[layer_name] = tf.nn.relu(input_layer, name=layer_name)

    def build_pool(self, layer_name, input_layer):
        self.net[layer_name] = tf.nn.max_pool(
            input_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
            padding='SAME', name=layer_name)

    def build_drop(self, layer_name, input_layer, drop_layer_name):
        drop_layer = self.net[drop_layer_name]
        self.net[layer_name] = tf.nn.dropout(
            input_layer, drop_layer, name=layer_name)

    def build_deconv(self, layer_name, input_layer, shape, strides,
                     padding='VALID'):
        weight_name = layer_name + '_weight'
        weight = tf.get_variable(weight_name, shape=shape)
        self.net[weight_name] = weight

        b = tf.shape(input_layer)[0]
        h = tf.shape(input_layer)[1]
        w = tf.shape(input_layer)[2]
        d = tf.shape(input_layer)[3]
        _, sh, sw, _ = strides
        kh, kw, _, _ = shape

        output_shape = tf.pack([b, sh * (h - 1) + kh, sw * (w - 1) + kw, d])

        self.net[layer_name] = tf.nn.conv2d_transpose(
            input_layer, weight, output_shape, strides=strides,
            padding=padding, name=layer_name)

    def build_slice(self, layer_name, input_layer, begin, shape_layer_name):
        size = tf.shape(self.net[shape_layer_name])
        self.net[layer_name] = tf.slice(input_layer, begin, size)

    def build_layer(self, layer_type, layer_name, input_layer_name, **kwargs):
        input_layer = self.net[input_layer_name]

        if layer_type == 'conv':
            self.build_conv(layer_name, input_layer, **kwargs)
        elif layer_type == 'relu':
            self.build_relu(layer_name, input_layer, **kwargs)
        elif layer_type == 'pool':
            self.build_pool(layer_name, input_layer, **kwargs)
        elif layer_type == 'drop':
            self.build_drop(layer_name, input_layer, **kwargs)
        elif layer_type == 'deconv':
            self.build_deconv(layer_name, input_layer, **kwargs)
        elif layer_type == 'slice':
            self.build_slice(layer_name, input_layer, **kwargs)

    def set_default_value(self, sess, caffe_mat, layer_id_map):
        for layer_name, idxs in layer_id_map.items():
            idx, bias_term = idxs

            weight = caffe_mat[idx][1][0].transpose((2, 3, 1, 0))
            if bias_term:
                bias = caffe_mat[idx][1][1]

            if layer_name.startswith('upscore'):
                weight = weight[:, :, :self.output_dim, :self.output_dim]
                bias = bias[:self.output_dim]

            if layer_name.startswith('score'):
                weight = weight[:, :, :, :self.output_dim]
                bias = bias[:self.output_dim]

            name = layer_name + '_weight'
            sess.run(tf.assign(self.net[name], weight))

            if bias_term:
                name = layer_name + '_bias'
                sess.run(tf.assign(self.net[name], bias))
