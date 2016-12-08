import caffe
import numpy as np


MODEL_DEF = 'deploy.prototxt'
MODEL_WEIGHT = 'fcn8s-heavy-pascal.caffemodel'
MAT_RESULT = 'fcn8s-heavy-pascal.mat'


def main():
    net = caffe.Net(MODEL_DEF, MODEL_WEIGHT, caffe.TRAIN)

    mat = []
    for i in range(len(net.layers)):
        mat_type = net.layers[i].type
        mat_data = []
        for j in range(len(net.layers[i].blobs)):
            mat_data.append(net.layers[i].blobs[j].data)
        mat.append((mat_type, mat_data))

    dt = np.dtype([('type', np.str_, 16), ('data', np.ndarray)])
    results = np.array(mat, dtype=dt)
    results.dump(MAT_RESULT)


if __name__ == '__main__':
    main()
