from scipy import misc, signal, stats
import matplotlib.pyplot as plt 
import random
import numpy as np

template_start = (10, 20)
template_end = (10,20)

img = np.array(misc.imresize(misc.imread('stop_sign.jpg', mode = 'L'), (200, 250)), dtype="float32")
img = img / img.max()
template = img[15:55, 50:80]


template = template - template.max() / 2


template_matching = signal.correlate(img, template)

print img.shape, template_matching.shape, template.shape


plt.subplot(3,1,1)
plt.imshow(img, cmap="Greys")
plt.subplot(3,1,2)
plt.imshow(template, interpolation="none")
plt.subplot(3,1,3)
plt.imshow(template_matching)
plt.show()




