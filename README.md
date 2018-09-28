# TensorFlow implementation of Automatic Portrait Segmentation

## Setup

### Dependencies
- Python 3.6+
- [TensorFlow](https://github.com/tensorflow/tensorflow) 1.6+
- [caffe](https://github.com/BVLC/caffe) (only use to transform pretrained model)

### Prepare environment, model and data

```shell
$ pip install -r requirements.txt
$ wget http://dl.caffe.berkeleyvision.org/fcn8s-heavy-pascal.caffemodel
$ python caffe_mat_transform.py
$ mkdir model
$ cd data
$ python data_download.py
```

## Usage

### Training

```shell
$ python train.py
```

### Testing
```shell
$ python test.py
```

## References

- [Fully Convolutional Models for Semantic Segmentation](https://arxiv.org/abs/1605.06211) by Evan Shelhamer*, Jonathan Long*, Trevor Darrell
- [Automatic Portrait Segmentation for Image Stylization](http://xiaoyongshen.me/webpage_portrait/index.html) by Xiaoyong Shen, Aaron Hertzmann, Jiaya Jia, Sylvain Paris, Brian Price, Eli Shechtman, Ian Sachs
