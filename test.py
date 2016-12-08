import numpy as np
import scipy.misc
import tensorflow as tf

from net import FCN8s


COLOR_SET = [
    [255, 255, 255], [125, 135, 185], [190, 193, 212], [214, 188, 192],
    [187, 119, 132], [142, 6, 59], [74, 111, 227], [133, 149, 225],
    [181, 187, 227], [230, 175, 185], [224, 123, 145], [211, 63, 106],
    [17, 198, 56], [141, 213, 147], [198, 222, 199], [234, 211, 198],
    [240, 185, 141], [239, 151, 8], [15, 207, 192], [156, 222, 214],
    [213, 234, 231], [243, 225, 235], [246, 196, 225], [247, 156, 212]
]


def build_image(filename):
    MEAN_VALUES = np.array([104.00698793, 116.66876762, 122.67891434])
    MEAN_VALUES = MEAN_VALUES.reshape((1, 1, 1, 3))
    img = scipy.misc.imread(filename, mode='RGB')[:, :, ::-1]
    height, width, _ = img.shape
    img = np.reshape(img, (1, height, width, 3)) - MEAN_VALUES
    return img


def save_image(result, filename):
    s = set()
    _, h, w = result.shape
    result = result.reshape(h*w)
    image = []
    for v in result:
        image.append(COLOR_SET[v])
        if v not in s:
            s.add(v)
    image = np.array(image)
    image = np.reshape(image, (h, w, 3))
    scipy.misc.imsave(filename, image)


def test(net, image_name):
    image = build_image(image_name)

    with tf.Session() as sess:
        saver = tf.train.Saver(tf.all_variables())
        model_file = tf.train.latest_checkpoint('./model/')
        if model_file:
            saver.restore(sess, model_file)
        else:
            raise Exception('Testing needs pre-trained model!')

        feed_dict = {
            net['image']: image,
            net['drop_rate']: 1
        }
        result = sess.run(tf.argmax(net['score'], dimension=3),
                          feed_dict=feed_dict)
    return result


if __name__ == '__main__':
    fcn = FCN8s(2)
    save_image(test(fcn.net, 'image.png'), 'result.png')
