import tensorflow as tf
import numpy as np
import scipy.misc


COLOR_SET = [
    [255, 255, 255], [125, 135, 185], [190, 193, 212], [214, 188, 192],
    [187, 119, 132], [142, 6, 59], [74, 111, 227], [133, 149, 225],
    [181, 187, 227], [230, 175, 185], [224, 123, 145], [211, 63, 106],
    [17, 198, 56], [141, 213, 147], [198, 222, 199], [234, 211, 198],
    [240, 185, 141], [239, 151, 8], [15, 207, 192], [156, 222, 214],
    [213, 234, 231], [243, 225, 235], [246, 196, 225], [247, 156, 212]
]


def build_fcn8s(caffe_mat, image):
    net = {}
    _, h, w, _ = image.shape
    DROPOUT_RATIO = 1  # 0.5 for learning

    net['input'] = tf.Variable(tf.zeros([1, h, w, 3]))
    net['padded_input'] = tf.pad(net['input'],
                                 [[0, 0], [99, 99], [99, 99], [0, 0]])

    net['conv1_1'] = conv_layer('conv1_1', net['padded_input'],
                                get_weight(caffe_mat, 2),
                                get_bias(caffe_mat, 2))
    net['relu1_1'] = relu_layer('relu1_1', net['conv1_1'])
    net['conv1_2'] = conv_layer('conv1_2', net['relu1_1'],
                                get_weight(caffe_mat, 4),
                                get_bias(caffe_mat, 4))
    net['relu1_2'] = relu_layer('relu1_2', net['conv1_2'])
    net['pool1'] = pool_layer('pool1', net['relu1_2'])

    net['conv2_1'] = conv_layer('conv2_1', net['pool1'],
                                get_weight(caffe_mat, 7),
                                get_bias(caffe_mat, 7))
    net['relu2_1'] = relu_layer('relu2_1', net['conv2_1'])
    net['conv2_2'] = conv_layer('conv2_2', net['relu2_1'],
                                get_weight(caffe_mat, 9),
                                get_bias(caffe_mat, 9))
    net['relu2_2'] = relu_layer('relu2_2', net['conv2_2'])
    net['pool2'] = pool_layer('pool2', net['relu2_2'])

    net['conv3_1'] = conv_layer('conv3_1', net['pool2'],
                                get_weight(caffe_mat, 12),
                                get_bias(caffe_mat, 12))
    net['relu3_1'] = relu_layer('relu3_1', net['conv3_1'])
    net['conv3_2'] = conv_layer('conv3_2', net['relu3_1'],
                                get_weight(caffe_mat, 14),
                                get_bias(caffe_mat, 14))
    net['relu3_2'] = relu_layer('relu3_2', net['conv3_2'])
    net['conv3_3'] = conv_layer('conv3_3', net['relu3_2'],
                                get_weight(caffe_mat, 16),
                                get_bias(caffe_mat, 16))
    net['relu3_3'] = relu_layer('relu3_3', net['conv3_3'])
    net['pool3'] = pool_layer('pool3', net['relu3_3'])

    net['conv4_1'] = conv_layer('conv4_1', net['pool3'],
                                get_weight(caffe_mat, 20),
                                get_bias(caffe_mat, 20))
    net['relu4_1'] = relu_layer('relu4_1', net['conv4_1'])
    net['conv4_2'] = conv_layer('conv4_2', net['relu4_1'],
                                get_weight(caffe_mat, 22),
                                get_bias(caffe_mat, 22))
    net['relu4_2'] = relu_layer('relu4_2', net['conv4_2'])
    net['conv4_3'] = conv_layer('conv4_3', net['relu4_2'],
                                get_weight(caffe_mat, 24),
                                get_bias(caffe_mat, 24))
    net['relu4_3'] = relu_layer('relu4_3', net['conv4_3'])
    net['pool4'] = pool_layer('pool4', net['relu4_3'])

    net['conv5_1'] = conv_layer('conv5_1', net['pool4'],
                                get_weight(caffe_mat, 28),
                                get_bias(caffe_mat, 28))
    net['relu5_1'] = relu_layer('relu5_1', net['conv5_1'])
    net['conv5_2'] = conv_layer('conv5_2', net['relu5_1'],
                                get_weight(caffe_mat, 30),
                                get_bias(caffe_mat, 30))
    net['relu5_2'] = relu_layer('relu5_2', net['conv5_2'])
    net['conv5_3'] = conv_layer('conv5_3', net['relu5_2'],
                                get_weight(caffe_mat, 32),
                                get_bias(caffe_mat, 32))
    net['relu5_3'] = relu_layer('relu5_3', net['conv5_3'])
    net['pool5'] = pool_layer('pool5', net['relu5_3'])

    net['fc6'] = conv_layer('fc6', net['pool5'],
                            get_weight(caffe_mat, 35),
                            get_bias(caffe_mat, 35), 'VALID')
    net['relu6'] = relu_layer('relu6', net['fc6'])
    net['drop6'] = tf.nn.dropout(net['relu6'], DROPOUT_RATIO)
    net['fc7'] = conv_layer('fc7', net['drop6'],
                            get_weight(caffe_mat, 37),
                            get_bias(caffe_mat, 37))
    net['relu7'] = relu_layer('relu7', net['fc7'])
    net['drop7'] = tf.nn.dropout(net['relu7'], DROPOUT_RATIO)

    net['score_fr'] = conv_layer('score_fr', net['drop7'],
                                 get_weight(caffe_mat, 39),
                                 get_bias(caffe_mat, 39))

    b, h, w, d = net['score_fr'].get_shape()
    output_shape = tf.pack([1, h + h + 2, w + w + 2, d])
    net['upscore2'] = tf.nn.conv2d_transpose(net['score_fr'],
                                             get_weight(caffe_mat, 40),
                                             output_shape,
                                             strides=[1, 2, 2, 1],
                                             padding='VALID')

    net['score_pool4'] = conv_layer('score_pool4', net['pool4'],
                                    get_weight(caffe_mat, 42),
                                    get_bias(caffe_mat, 42))
    net['score_pool4c'] = tf.slice(net['score_pool4'], [0, 5, 5, 0],
                                   net['upscore2'].get_shape())
    net['fuse_pool4'] = net['upscore2'] + net['score_pool4c']

    b, h, w, d = net['fuse_pool4'].get_shape()
    output_shape = tf.pack([1, h + h + 2, w + w + 2, d])
    net['upscore_pool4'] = tf.nn.conv2d_transpose(net['fuse_pool4'],
                                                  get_weight(caffe_mat, 45),
                                                  output_shape,
                                                  strides=[1, 2, 2, 1],
                                                  padding='VALID')

    net['score_pool3'] = conv_layer('score_pool3', net['pool3'],
                                    get_weight(caffe_mat, 47),
                                    get_bias(caffe_mat, 47))
    net['score_pool3c'] = tf.slice(net['score_pool3'], [0, 9, 9, 0],
                                   net['upscore_pool4'].get_shape())
    net['fuse_pool3'] = net['upscore_pool4'] + net['score_pool3c']

    b, h, w, d = net['fuse_pool3'].get_shape()
    output_shape = tf.pack([1, (h + 1) * 8, (w + 1) * 8, d])
    net['upscore8'] = tf.nn.conv2d_transpose(net['fuse_pool3'],
                                             get_weight(caffe_mat, 50),
                                             output_shape,
                                             strides=[1, 8, 8, 1],
                                             padding='VALID')

    _, h, w, _ = net['input'].get_shape()
    b, _, _, d = net['upscore8'].get_shape()
    output_shape = tf.pack([b, h, w, d])
    net['score'] = tf.slice(net['upscore8'], [0, 31, 31, 0], output_shape)

    return net


def conv_layer(layer_name, layer_input, weight, bias, padding='SAME'):
    return tf.nn.bias_add(
        tf.nn.conv2d(layer_input, weight, strides=[1, 1, 1, 1],
                     padding=padding, name=layer_name), bias)


def relu_layer(layer_name, layer_input):
    return tf.nn.relu(layer_input, name=layer_name)


def pool_layer(layer_name, layer_input):
    return tf.nn.max_pool(
        layer_input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
        padding='SAME', name=layer_name)


def get_weight(mat, idx):
    return mat[idx][1][0].transpose((2, 3, 1, 0))


def get_bias(mat, idx):
    return mat[idx][1][1]


def build_image(filename):
    MEAN_VALUES = np.array([104.00698793, 116.66876762, 122.67891434])
    MEAN_VALUES = MEAN_VALUES.reshape((1, 1, 1, 3))

    img = scipy.misc.imread(filename, mode='RGB')[:, :, ::-1]
    height, width, _ = img.shape
    img = np.reshape(img, (1, height, width, 3)) - MEAN_VALUES
    return img


def save_image(result, filename='result.png'):
    s = set()
    _, h, w = result.shape
    result = result.reshape(h*w)
    image = []
    for v in result:
        image.append(COLOR_SET[v])
        if v not in s:
            s.add(v)
    image = np.array(image, dtype=np.uint8)
    image = np.reshape(image, (h, w, 3))
    scipy.misc.imsave(filename, image)


def main():
    model_filename = '../fcn8s-heavy-pascal.mat'
    input_image_filename = '../cat.jpg'

    caffe_mat = np.load(model_filename)
    image = build_image(input_image_filename)
    net = build_fcn8s(caffe_mat, image)
    feed_dict = {
        net['input']: image
    }

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        result = sess.run(tf.argmax(net['score'], dimension=3),
                          feed_dict=feed_dict)

    save_image(result)


if __name__ == '__main__':
    main()
