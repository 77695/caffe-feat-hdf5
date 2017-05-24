import matplotlib.pyplot as plt
import h5py
import numpy as np

all_filter=np.load('./feat/filter_conv1.npy')

lns=all_filter.shape[0]/8+1
for j in xrange(all_filter.shape[0]):
  plt.subplot(lns,8,j+1)
  plt.axis('off')
  plt.imshow(all_filter[j,0],cmap='gray',interpolation='nearest')
plt.savefig('filter.png')
plt.close()
# plt.show()
