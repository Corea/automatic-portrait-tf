from pathlib import Path
import random

import numpy as np
import scipy.io
import scipy.misc
import tensorflow as tf
import tensorflow.contrib.slim as slim

import slim_net


IMAGE_DIR = Path("./data/images_data_crop")
MASK_DIR = Path("./data/images_mask")
BATCH_SIZE = 2
HEIGHT = 800
WIDTH = 600
LEARNING_RATE = 1e-4
NUM_CLASSES = 2

PRETRAINED_MODEL = "fcn8s-heavy-pascal.mat"
LAYER_ID_MAP = {
    'conv1/conv1_1': [2, True],
    'conv1/conv1_2': [4, True],

    'conv2/conv2_1': [7, True],
    'conv2/conv2_2': [9, True],

    'conv3/conv3_1': [12, True],
    'conv3/conv3_2': [14, True],
    'conv3/conv3_3': [16, True],

    'conv4/conv4_1': [20, True],
    'conv4/conv4_2': [22, True],
    'conv4/conv4_3': [24, True],

    'conv5/conv5_1': [28, True],
    'conv5/conv5_2': [30, True],
    'conv5/conv5_3': [32, True],

    'fc6': [35, True],
    'fc7': [37, True],

    'score_fr': [39, True],

    'upscore2': [40, False],
    'score_pool4': [42, True],

    'upscore_pool4': [45, False],
    'score_pool3': [47, True],

    'upscore8': [50, False],
}


def assign_default_value(sess, output_dim):
    caffe_mat = np.load(PRETRAINED_MODEL, encoding="latin1")
    graph = tf.get_default_graph()

    for layer_name, idxs in LAYER_ID_MAP.items():
        idx, bias_term = idxs

        weight = caffe_mat[idx][1][0].transpose((2, 3, 1, 0))
        if bias_term:
            bias = caffe_mat[idx][1][1]

        if layer_name.startswith('upscore'):
            weight = weight[:, :, :output_dim, :output_dim]
            bias = bias[:output_dim]

        if layer_name.startswith('score'):
            weight = weight[:, :, :, :output_dim]
            bias = bias[:output_dim]

        weight_tensor_name = f"fcn8s/{layer_name}/weights:0"
        sess.run(tf.assign(graph.get_tensor_by_name(weight_tensor_name), weight))

        if bias_term:
            bias_tensor_name = f"fcn8s/{layer_name}/biases:0"
            sess.run(tf.assign(graph.get_tensor_by_name(bias_tensor_name), bias))


def build_image(filename):
    MEAN_VALUES = np.array([104.00698793, 116.66876762, 122.67891434])
    MEAN_VALUES = MEAN_VALUES.reshape((1, 1, 1, 3))
    img = scipy.misc.imread(filename, mode="RGB")[:, :, ::-1]
    height, width, _ = img.shape
    img = np.reshape(img, (1, height, width, 3)) - MEAN_VALUES
    return img


def train():
    inputs = tf.placeholder(tf.float32, [BATCH_SIZE, HEIGHT, WIDTH, 3])
    with slim.arg_scope(slim_net.fcn8s_arg_scope()):
        logits, _ = slim_net.fcn8s(inputs, NUM_CLASSES)

        label = tf.placeholder(tf.uint8, shape=[BATCH_SIZE, HEIGHT, WIDTH])
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=tf.reshape(logits, [-1, 2]),
            labels=tf.stop_gradient(tf.one_hot(tf.reshape(label, [-1]), NUM_CLASSES))))

    with tf.Session() as sess:
        global_step = tf.Variable(0, name="global_step", trainable=False)

        saver = tf.train.Saver(tf.global_variables())
        model_file = tf.train.latest_checkpoint("./model/")
        if model_file:
            print(f"Restore from {model_file}")
            saver.restore(sess, model_file)
        else:
            print("Initialize from pre-trained model")
            sess.run(tf.global_variables_initializer())
            assign_default_value(sess, NUM_CLASSES)

        print("Start to train")
        optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
        train_op = optimizer.minimize(cost, global_step=global_step)

        all_images = list(IMAGE_DIR.glob("*.jpg"))
        while True:
            image_mat = []
            label_mat = []
            images = random.sample(all_images, BATCH_SIZE)
            for image_fullpath in images:
                mask_filename = f"{image_fullpath.stem}_mask.mat"
                mask_fullpath = MASK_DIR / mask_filename

                image_mat.append(build_image(image_fullpath))
                label_mat.append(scipy.io.loadmat(mask_fullpath)["mask"])

            feed_dict = {
                inputs: np.concatenate(image_mat),
                label: np.stack(label_mat)
            }
            _, loss, step = sess.run([train_op, cost, global_step], feed_dict=feed_dict)

            if step % 1 == 0:
                print(f"[Step {step}] Loss: {loss}")
            if step % 500 == 0:
                saver.save(sess, "./model/PortraitFCN", global_step=step)
                print(f"[Step {step}] Saved")
            if step >= 100000:
                break


if __name__ == "__main__":
    train()
