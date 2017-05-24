import sys
sys.path.append('/home/heanfeng/caffe/python')
import caffe
import numpy as np
MODEL_FILE = '/home/heanfeng/DeepLearning/caffe-feat-hdf5/lenet_train_test_2.prototxt'
PRETRAINED_MODEL = '/home/heanfeng/DeepLearning/caffe-feat-hdf5/lenet_iter_10000.caffemodel'

net=caffe.Net(MODEL_FILE, PRETRAINED_MODEL,caffe.TEST)

for layer_name,param in net.params.iteritems():
    print layer_name,param[0].data.shape
    np.save('./feat/filter_'+layer_name+'.npy',param[0].data)