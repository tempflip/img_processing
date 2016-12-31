from scipy import misc, signal, stats
import matplotlib.pyplot as plt 
import random
import numpy as np
from skimage import filters


def gnoise(image, sigma = 30):
	img = np.array(image)
	for i, row in enumerate(img):
		for (j, pixel) in enumerate(row):
			img[i,j] = img[i,j] + random.gauss(0, sigma)
	return img

def g_blur(image, n=10):
	img = np.zeros(im.shape)
	mask = np.ones((n,n))
	
	original_max = image.max()

	for i, row in enumerate(im):
		if i == 0 or i == len(im)-1 : continue
		for j, pixel in enumerate(row):
			if i == 0 or i == len(row)-1 : continue

			ii = mask.shape[0]
			jj = mask.shape[1]

			window = image[i:i+ii, j:j+jj]

			w_sum = sum(sum(window))
			blurred = w_sum / window.shape[0] * window.shape[1]

			img[i,j] = blurred

	print original_max , img.max(), img.max() / original_max

	return img

def gaussian_dist(m, sigma, size):
	pdf = stats.norm(m, sigma).pdf
	linspace = np.linspace(- sigma, sigma, size)
	dist = np.array([[pdf(x) * pdf(y) for x in linspace] for y in linspace])
	return dist

def unsharp_filter(n = 10, b = 1, c = 2):
	#gaussian = gaussian_dist(0, 1, n)
	uniform_blur = np.array([[b + 0.0 for x in range(n)] for y in range(n)])
	uniform_blur = uniform_blur / (n * 2)

	inpulse = np.array([[0 for x in range(n)] for y in range(n)])
	inpulse[n/2][n/2] = c

	
	return inpulse - uniform_blur


FNAME1 = 'cat1.jpg'
FNAME2 = 'solidYellowCurve2.jpg'
W = 500
H = 200

img1 = misc.imread(FNAME1, mode='L')
img2 = misc.imread(FNAME2, mode='L')

img1 = misc.imresize(img1, (H, W))
img2 = misc.imresize(img2, (H, W))

#im = gnoise(img1, sigma = 0)

my_filter = np.array([[1,1,1,1,1],
					[1,1,1,1,1],
					[1,1,1,1,1],
					[1,1,1,1,1],
					[1,1,1,1,1]])

im = signal.correlate(img2, my_filter)
plt.subplot(2,1,1)
plt.imshow(img2)
plt.subplot(2,1,2)
plt.imshow(im)
plt.show()
exit()






my_filter_gaussian = gaussian_dist(0, 1, 10)
my_unsharp_filter =  unsharp_filter(3, 1, 2.5)
print my_unsharp_filter

im = img1
#im2 = filters.gaussian(im, sigma=5)
#im2 = signal.correlate(im, my_filter_gaussian)
#im2 = signal.convolve(im, my_filter)
im2 = signal.correlate(im, my_unsharp_filter)

plt.subplot(2,1,1)
plt.imshow(im, cmap="Greys")
plt.subplot(2,1,2)
plt.imshow(im2, cmap="Greys")
#plt.imshow(g_blur(im))

"""
plt.subplot(5, 1, 1)
plt.imshow(gnoise(img1, sigma = 0))
plt.subplot(5, 1, 2)
plt.imshow(gnoise(img1, sigma = 10))
plt.subplot(5, 1, 3)
plt.imshow(gnoise(img1, sigma = 20))
plt.subplot(5, 1, 4)
plt.imshow(gnoise(img1, sigma = 40))
plt.subplot(5, 1, 5)
plt.imshow(gnoise(img1, sigma = 80))
"""



plt.show()
