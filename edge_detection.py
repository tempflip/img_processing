from scipy import misc, signal, stats
import matplotlib.pyplot as plt 
import random
import numpy as np
from skimage.filters.rank import gradient
from skimage.filter import roberts, sobel, scharr, prewitt
import time
from math import *


def gaussian_dist(m, sigma, size):
	pdf = stats.norm(m, sigma).pdf
	linspace = np.linspace(- sigma, sigma, size)
	dist = np.array([[pdf(x) * pdf(y) for x in linspace] for y in linspace])
	return dist

img = np.array(misc.imresize(misc.imread('solidYellowCurve2.jpg', mode = 'L'), (400, 500)), dtype="float32")
img = img / img.max()
#img = img - img.max()/2



kernel_der = np.array([[-1, 1]])

kernel_roberts = np.array([[1,0]
				  ,[0,-1]
					])
kernel_sobel = np.array([
	[-1, 0, 1],
	[-2, 0, 2],
	[-1, 0, 1]
	])


kernel_prewitt = np.array([
	[-1, 0, 1],
	[-1, 0, 1],
	[-1, 0, 1]
	])



my_kernel1 = kernel_sobel

g = gaussian_dist(0, 5, 3)
kernel = signal.correlate(g, kernel_sobel)



out = np.sqrt(signal.correlate(img, kernel) ** 2 + signal.correlate(img, kernel.T) ** 2)
misc.imsave("edge1.png", out)

#plt.imshow(out)

"""
g = gaussian_dist(0, 5, 10)
my_kernel3 = signal.correlate(g, kernel_sobel)

g = gaussian_dist(0, 5, 15)
my_kernel4 = signal.correlate(g, kernel_sobel)





out1 = np.sqrt(signal.correlate(img, my_kernel1)**2 + signal.correlate(img, my_kernel1.T)**2)
out2 = np.sqrt(signal.correlate(img, my_kernel2)**2 + signal.correlate(img, my_kernel2.T)**2)
out3 = np.sqrt(signal.correlate(img, my_kernel3)**2 + signal.correlate(img, my_kernel3.T)**2)
out4 = np.sqrt(signal.correlate(img, my_kernel4)**2 + signal.correlate(img, my_kernel4.T)**2)




plt.subplot(4,1,1)
plt.imshow(out1, interpolation="none")
plt.subplot(4,1,2)
plt.imshow(out2, interpolation="none")
plt.subplot(4,1,3)
plt.imshow(out3, interpolation="none")
plt.subplot(4,1,4)
plt.imshow(out4, interpolation="none")
"""
#plt.show()




exit()






