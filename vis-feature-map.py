import matplotlib.pyplot as plt
import h5py
import numpy as np

with h5py.File('./feat/conv2_mnist_out_0','r') as fs:
  all_featmap=np.array(fs['data'])
  for i in xrange(0,5):
    print 'label',int(np.array(fs['label'][i][0,0,0]))
    lns=all_featmap.shape[1]/8+1
    for j in xrange(all_featmap.shape[1]):
      plt.subplot(lns,8,j+1)
      plt.axis('off')
      plt.imshow(all_featmap[i,j],cmap='gray',interpolation='nearest')
    plt.savefig('feat_'+str(int(np.array(fs['label'][i][0,0,0])))+'.jpg')
    plt.close()
    # plt.show()
