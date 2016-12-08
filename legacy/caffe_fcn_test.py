import caffe
import numpy as np


def load_caffe_model():
    model_def = 'deploy.prototxt'
    model_weight = 'fcn8s-heavy-pascal.caffemodel'
    return caffe.Net(model_def, model_weight, caffe.TEST)


def main():
    net = load_caffe_model()

    image = caffe.io.load_image('cat.jpg')
    h, w, d = image.shape

    net.blobs['data'].reshape(1, d, h, w)
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

    mu = np.array([104.00698793, 116.66876762, 122.67891434])
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_raw_scale('data', 255)
    transformer.set_mean('data', mu)
    transformer.set_channel_swap('data', (2, 1, 0))

    transformed_image = transformer.preprocess('data', image)
    net.blobs['data'].data[...] = transformed_image
    net.forward()

    result = net.blobs['score'].data[0].argmax(axis=0)
    return result


if __name__ == '__main__':
    caffe.set_device(0)
    caffe.set_mode_gpu()
    main()
