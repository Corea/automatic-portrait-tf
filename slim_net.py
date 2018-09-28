import tensorflow as tf
import tensorflow.contrib.slim as slim


def fcn8s_arg_scope(weight_decay=0.0005):
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        biases_initializer=tf.zeros_initializer()):
        with slim.arg_scope([slim.conv2d, slim.max_pool2d], padding="SAME"):
            with slim.arg_scope([slim.conv2d_transpose], padding="VALID",
                                biases_initializer=None) as arg_sc:
                return arg_sc


def fcn8s(inputs,
          num_classes,
          is_training=True,
          dropout_keep_prob=0.5,
          scope="fcn8s"):

    with tf.variable_scope(scope, "fcn8s", [inputs]) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        net = tf.pad(inputs, [[0, 0], [99, 99], [99, 99], [0, 0]], name="pad_layer")

        with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                            outputs_collections=end_points_collection):
            # Encoder
            net = slim.repeat(net, 2, slim.conv2d, 64, [3, 3], scope="conv1")
            net = slim.max_pool2d(net, [2, 2], scope="pool1")

            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope="conv2")
            net = slim.max_pool2d(net, [2, 2], scope="pool2")

            net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope="conv3")
            net = pool3 = slim.max_pool2d(net, [2, 2], scope="pool3")

            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope="conv4")
            net = pool4 = slim.max_pool2d(net, [2, 2], scope="pool4")

            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope="conv5")
            net = slim.max_pool2d(net, [2, 2], scope="pool5")

            net = slim.conv2d(net, 4096, [7, 7], padding="VALID", scope="fc6")
            net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope="drop6")

            net = slim.conv2d(net, 4096, [1, 1], scope="fc7")
            net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope="drop7")

            net = slim.conv2d(net, num_classes, [1, 1], scope="score_fr")

            # Decoder
            upscore2 = slim.conv2d_transpose(net, num_classes, [4, 4], stride=2, scope="upscore2")
            score_pool4 = slim.conv2d(pool4, num_classes, [1, 1], scope="score_pool4")
            score_pool4c = tf.slice(score_pool4, [0, 5, 5, 0], tf.shape(upscore2),
                                    name="score_pool4c")
            net = tf.add(upscore2, score_pool4c, name="fuse_pool4")

            upscore_pool4 = slim.conv2d_transpose(net, num_classes, [4, 4], stride=2,
                                                  scope="upscore_pool4")
            score_pool3 = slim.conv2d(pool3, num_classes, [1, 1], scope="score_pool3")
            score_pool3c = tf.slice(score_pool3, [0, 9, 9, 0], tf.shape(upscore_pool4),
                                    name="score_pool3c")
            net = tf.add(upscore_pool4, score_pool3c, name="fuse_pool3")

            net = slim.conv2d_transpose(net, num_classes, [16, 16], stride=8, scope="upscore8")

            b = tf.shape(inputs)[0]
            h = tf.shape(inputs)[1]
            w = tf.shape(inputs)[2]
            output_shape = tf.stack([b, h, w, num_classes])
            net = tf.slice(net, [0, 31, 31, 0], output_shape)

            end_points = slim.utils.convert_collection_to_dict(end_points_collection)

    return net, end_points
